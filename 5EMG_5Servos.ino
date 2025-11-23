/*
  MyoMimic – Adaptive kNN with Per-Gesture EMG Weighting (k = 5) + Kalman Filtering + High Sensitivity
  --------------------------------------------------------------------------------------------------
  - EMG:    GPIO25, 26, 27, 32, 33
  - Servos: GPIO13, 14, 15, 16, 17
  - LED:    GPIO2
  - Gestures: Relax, Thumb, Index, Middle, Ring, Little, Fist, Pinch, Peace
  - Type 'r' in Serial Monitor to restart calibration
*/

#include <ESP32Servo.h>
#include "EMGFilters.h"
#include <math.h>

// ---------------- Pin Setup ----------------
#define NUM_SENSORS 5
#define NUM_GESTURES 9
#define SAMPLES_PER_GESTURE 3

int EMG_PINS[NUM_SENSORS]   = {25, 26, 27, 32, 33};
int SERVO_PINS[NUM_SENSORS] = {13, 14, 15, 16, 17};
#define LED_PIN 2

// ---------------- Simple 1D Kalman Filter ----------------
struct Kalman1D {
  float x;  // state estimate (filtered value)
  float P;  // estimate covariance
  float Q;  // process noise (how fast we allow the value to change)
  float R;  // measurement noise (how noisy the measurements are)

  void init(float q, float r, float initialValue = 0.0f) {
    Q = q;
    R = r;
    x = initialValue;
    P = 1.0f; // initial uncertainty
  }

  float update(float z) {
    // Prediction: x = x (constant model), P = P + Q
    P = P + Q;

    // Measurement update
    float K = P / (P + R); // Kalman gain
    x = x + K * (z - x);
    P = (1.0f - K) * P;

    return x;
  }
};

// ---------------- Objects ----------------
Servo servos[NUM_SENSORS];
EMGFilters filters[NUM_SENSORS];
Kalman1D kf[NUM_SENSORS];

// ---------------- Constants ----------------
// From EMGFilters library
const int   SAMPLE_RATE    = SAMPLE_FREQ_1000HZ;
const int   HUM_NOTCH      = NOTCH_FREQ_50HZ;

// Sensitivity settings
const float GAIN           = 2.0f;       // increased EMG amplification (was 1.3)
const float BASELINE_ALPHA = 0.005f;     // more sensitive baseline averaging (smaller = lower baseline)
const float KALMAN_Q       = 0.15f;      // EMG dynamics
const float KALMAN_R       = 1.8f;       // trust measurement more → more sensitivity
const int   HOLD_TIME_MS   = 2000;       // hold time per calibration sample
const int   UPDATE_MS      = 300;        // main loop update period
const int   K              = 5;          // neighbours for kNN

// ---------------- Calibration / Runtime Data ----------------
float baseline[NUM_SENSORS] = {0};
float trainingData[NUM_GESTURES][SAMPLES_PER_GESTURE][NUM_SENSORS] = {0};
float weightsPerGesture[NUM_GESTURES][NUM_SENSORS] = {0}; // adaptive weights per gesture
int servoPos[NUM_SENSORS] = {0};

// ---------------- Gesture Names ----------------
const char* gestureNames[NUM_GESTURES] = {
  "Relax", "Thumb", "Index", "Middle", "Ring", "Little", "Fist", "Pinch", "Peace"
};

// ---------------- Function Prototypes ----------------
void runCalibration();
int classifyGesture(float *current, float &confidence);
void moveServosForGesture(int gestureID);
void computeGestureWeights();

// ====================================================
// -------------------- SETUP -------------------------
// ====================================================

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  for (int i = 0; i < NUM_SENSORS; i++) {
    filters[i].init(SAMPLE_RATE, HUM_NOTCH, true, true, true);
    servos[i].attach(SERVO_PINS[i]);
    servos[i].write(0);

    // Kalman parameters (high sensitivity)
    kf[i].init(KALMAN_Q, KALMAN_R, 0.0f);
  }

  Serial.println("\nMyoMimic Adaptive kNN + Kalman (High Sensitivity) Initialising...");
  delay(500);
  runCalibration();
}

// ====================================================
// -------------------- MAIN LOOP ---------------------
// ====================================================

void loop() {
  static unsigned long lastUpd = 0;

  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'r' || c == 'R') {
      Serial.println("\nRecalibration requested...");
      runCalibration();
    }
  }

  if (millis() - lastUpd < (unsigned long)UPDATE_MS) return;
  lastUpd = millis();

  // --- Read, normalise and Kalman-filter all sensors (HIGH SENSITIVITY MODE) ---
  float current[NUM_SENSORS];
  for (int i = 0; i < NUM_SENSORS; i++) {

    // Raw ADC → EMG bandpass + rectification
    int raw = analogRead(EMG_PINS[i]);
    float env = filters[i].update(raw) * GAIN;   // amplified envelope

    // Normalise using more sensitive baseline
    float norm = (env / (baseline[i] + 1e-6f)) * 120.0f;  
    // ↑ 120 instead of 100 → ~20% extra sensitivity

    // Kalman filtering (trust measurement more → faster reaction)
    float filtered = kf[i].update(norm);

    // Store result for classification
    current[i] = filtered;
  }

  // --- Classify gesture ---
  float confidence = 0;
  int g = classifyGesture(current, confidence);

  Serial.print("Detected: ");
  Serial.print(gestureNames[g]);
  Serial.print(" | Confidence: ");
  Serial.println(confidence, 2);

  moveServosForGesture(g);

  // Slightly lower LED confidence threshold for responsiveness
  digitalWrite(LED_PIN, (confidence > 0.18f) ? HIGH : LOW);
}

// ====================================================
// ---------------- CLASSIFICATION --------------------
// ====================================================

int classifyGesture(float *current, float &confidence) {
  struct GestureDist {
    int gesture;
    float dist;
  } dists[NUM_GESTURES * SAMPLES_PER_GESTURE];

  int idx = 0;
  for (int g = 0; g < NUM_GESTURES; g++) {
    for (int s = 0; s < SAMPLES_PER_GESTURE; s++) {
      float d = 0;
      for (int i = 0; i < NUM_SENSORS; i++) {
        float diff = current[i] - trainingData[g][s][i];
        d += weightsPerGesture[g][i] * (diff * diff);
      }
      dists[idx].gesture = g;
      dists[idx].dist = sqrtf(d);
      idx++;
    }
  }

  // Simple bubble sort (dataset is tiny, so this is fine)
  for (int i = 0; i < idx - 1; i++) {
    for (int j = i + 1; j < idx; j++) {
      if (dists[j].dist < dists[i].dist) {
        GestureDist tmp = dists[i];
        dists[i] = dists[j];
        dists[j] = tmp;
      }
    }
  }

  // Majority vote among K nearest
  int counts[NUM_GESTURES] = {0};
  for (int i = 0; i < K; i++) {
    counts[dists[i].gesture]++;
  }

  int bestGesture = 0;
  for (int g = 1; g < NUM_GESTURES; g++) {
    if (counts[g] > counts[bestGesture]) bestGesture = g;
  }

  float d1 = dists[0].dist;
  float d2 = dists[1].dist;
  confidence = 1.0f - (d1 / (d2 + 1e-6f));

  return bestGesture;
}

// ====================================================
// ---------------- SERVO CONTROL ---------------------
// ====================================================

void moveServosForGesture(int g) {
  int targets[NUM_SENSORS] = {0, 0, 0, 0, 0};

  switch (g) {
    case 0:
      // Relax – all open
      break;
    case 1:
      targets[0] = 160;               // Thumb
      break;
    case 2:
      targets[1] = 160;               // Index
      break;
    case 3:
      targets[2] = 160;               // Middle
      break;
    case 4:
      targets[3] = 160;               // Ring
      break;
    case 5:
      targets[4] = 160;               // Little
      break;
    case 6:
      for (int i = 0; i < NUM_SENSORS; i++) targets[i] = 160; // Fist
      break;
    case 7:
      targets[0] = 160;               // Pinch (Thumb + Index)
      targets[1] = 160;
      break;
    case 8:
      targets[1] = 160;               // Peace (Index + Middle)
      targets[2] = 160;
      break;
  }

  // Smooth servo motion
  for (int i = 0; i < NUM_SENSORS; i++) {
    if (abs(servoPos[i] - targets[i]) > 2) {
      servoPos[i] = 0.85f * servoPos[i] + 0.15f * targets[i];
      servos[i].write(servoPos[i]);
    }
  }
}

// ====================================================
// ---------------- CALIBRATION -----------------------
// ====================================================

void runCalibration() {
  Serial.println("\n--- Calibration Starting ---");
  delay(800);

  // Reset Kalman filters before calibration (keeps training data consistent)
  for (int i = 0; i < NUM_SENSORS; i++) {
    kf[i].init(KALMAN_Q, KALMAN_R, 0.0f);
    baseline[i] = 0.0f;
  }

  // ---- Relax phase ----
  Serial.println("Keep your hand relaxed...");
  unsigned long t0 = millis();
  while (millis() - t0 < (unsigned long)HOLD_TIME_MS) {
    for (int i = 0; i < NUM_SENSORS; i++) {
      int raw = analogRead(EMG_PINS[i]);
      float env = filters[i].update(raw) * GAIN;
      baseline[i] = (1.0f - BASELINE_ALPHA) * baseline[i] + BASELINE_ALPHA * env;
    }
    digitalWrite(LED_PIN, HIGH);
    delay(5);
  }
  digitalWrite(LED_PIN, LOW);
  Serial.println("Relaxation baseline set.");

  // ---- Gesture recordings ----
  for (int g = 0; g < NUM_GESTURES; g++) {
    Serial.println();
    Serial.print("Perform gesture: ");
    Serial.println(gestureNames[g]);

    for (int s = 0; s < SAMPLES_PER_GESTURE; s++) {
      Serial.print("  Sample "); Serial.print(s + 1); Serial.println("/3");
      delay(1000);

      // Optionally reset Kalman for cleaner per-sample averaging
      for (int i = 0; i < NUM_SENSORS; i++) {
        kf[i].init(KALMAN_Q, KALMAN_R, 0.0f);
      }

      unsigned long start = millis();
      float accum[NUM_SENSORS] = {0};
      int count = 0;

      while (millis() - start < (unsigned long)HOLD_TIME_MS) {
        for (int i = 0; i < NUM_SENSORS; i++) {
          int raw = analogRead(EMG_PINS[i]);
          float env = filters[i].update(raw) * GAIN;
          float norm = (env / (baseline[i] + 1e-6f)) * 120.0f;

          float filtered = kf[i].update(norm);
          accum[i] += filtered;
        }
        count++;
        delay(5);
      }

      for (int i = 0; i < NUM_SENSORS; i++) {
        trainingData[g][s][i] = (count > 0) ? (accum[i] / count) : 0.0f;
      }

      Serial.println("Recorded");
    }
  }

  computeGestureWeights();
  Serial.println("\nCalibration complete. Adaptive weights applied.\n");
}

// ====================================================
// ---------------- WEIGHT COMPUTATION ----------------
// ====================================================

void computeGestureWeights() {
  for (int g = 0; g < NUM_GESTURES; g++) {
    float mean[NUM_SENSORS] = {0};

    // Mean activation per sensor for this gesture
    for (int s = 0; s < SAMPLES_PER_GESTURE; s++) {
      for (int i = 0; i < NUM_SENSORS; i++) {
        mean[i] += trainingData[g][s][i];
      }
    }
    for (int i = 0; i < NUM_SENSORS; i++) {
      mean[i] /= SAMPLES_PER_GESTURE;
    }

    // Weights proportional to activation above baseline
    float total = 0;
    for (int i = 0; i < NUM_SENSORS; i++) {
      weightsPerGesture[g][i] = fmaxf(0.0f, mean[i] - baseline[i]);
      total += weightsPerGesture[g][i];
    }

    if (total < 1e-6f) total = 1.0f;
    for (int i = 0; i < NUM_SENSORS; i++) {
      weightsPerGesture[g][i] /= total;
    }

    Serial.print("Gesture ");
    Serial.print(gestureNames[g]);
    Serial.print(" Weights: ");
    for (int i = 0; i < NUM_SENSORS; i++) {
      Serial.print(weightsPerGesture[g][i], 2);
      Serial.print(" ");
    }
    Serial.println();
  }
}
