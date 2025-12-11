/*
   MyoMimic – Pure k-NN (No Thresholds) – 4 EMG Sensors → 5 Servos
   --------------------------------------------------------------
   EMG layout:
        CH0: Bicep      (GPIO33)
        CH1: Forearm L  (GPIO25)
        CH2: Forearm M  (GPIO26)
        CH3: Forearm R  (GPIO27)

   Gestures:
        0: OpenPalm
        1: Fist
        2: Pinch
        3: Peace

   Feature vector (7 values):
        [bN, f0N, f1N, f2N, M0, M1, M2]
*/

// MAIN //

#include <Arduino.h>
#include <ESP32Servo.h>
#include "EMGFilters.h"
#include <math.h>

#define NUM_EMG 4
#define NUM_SERVOS 5
#define NUM_GESTURES 4
#define FEATURE_DIM 7
#define SAMPLES_PER_GESTURE 3

int EMG_PINS[NUM_EMG]      = {33, 25, 26, 27};
int SERVO_PINS[NUM_SERVOS] = {13, 14, 15, 16, 17};

const int SAMPLE_RATE = SAMPLE_FREQ_500HZ;
const int HUM_FREQ    = NOTCH_FREQ_50HZ;

const int   SERVO_MIN_ANGLE = 0;
const int   SERVO_MAX_ANGLE = 160;
const float SERVO_SMOOTH    = 0.2f;

const unsigned long CLASS_WINDOW_MS   = 150;
const unsigned long PRINT_INTERVAL_MS = 300;

// ---------------------------------- Kalman filter ----------------------------------
struct Kalman1D {
  float x, P, Q, R;
  void init(float q, float r, float initialValue = 0.0f) {
    Q = q; R = r;
    x = initialValue; P = 1.0f;
  }
  float update(float z) {
    P += Q;
    float K = P / (P + R);
    x = x + K * (z - x);
    P = (1.0f - K) * P;
    return x;
  }
};

const float KALMAN_Q = 200.0f;
const float KALMAN_R = 5000.0f;

// ---------------------------------- Globals ----------------------------------------
EMGFilters emgFilter[NUM_EMG];
Kalman1D   kf[NUM_EMG];
Servo      servos[NUM_SERVOS];

int servoPos[NUM_SERVOS] = {0};
int currentGesture = 0;

unsigned long lastSampleMicros = 0;
const unsigned long SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE;

float rmsAccum[NUM_EMG] = {0};
int   rmsCount = 0;
unsigned long lastFeatureTime = 0;
unsigned long lastPrintTime   = 0;

// k-NN storage: gesture × sample × feature dimension
float knnData[NUM_GESTURES][SAMPLES_PER_GESTURE][FEATURE_DIM];

const char* gestureNames[NUM_GESTURES] = {
  "OpenPalm", "Fist", "Pinch", "Peace"
};

// ---------- Function prototypes ----------
void runCalibration();
void gestureCalibration();
void computeRuntimeFeaturesAndClassify(const long *env);
void buildFeatureVector(const float rms[NUM_EMG], float out[FEATURE_DIM]);
int classifyGestureKNN(const float *feat);
void moveServosForGesture(int g);

// ====================================================
//                        SETUP
// ====================================================
void setup() {
  Serial.begin(115200);
  delay(200);

  Serial.println("\nMyoMimic – Pure k-NN Version (No Thresholds)");

  for (int i = 0; i < NUM_EMG; i++) {
    emgFilter[i].init(SAMPLE_RATE, HUM_FREQ, true, true, true);
    kf[i].init(KALMAN_Q, KALMAN_R);
  }

  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].setPeriodHertz(50);
    servos[i].attach(SERVO_PINS[i]);
    servos[i].write(0);
  }

  runCalibration();
  Serial.println("Calibration complete. System live.\n");
}

// ====================================================
//                         LOOP
// ====================================================
void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'r' || c == 'R') runCalibration();
  }

  unsigned long nowMicros = micros();
  if (nowMicros - lastSampleMicros < SAMPLE_PERIOD_US) return;
  lastSampleMicros = nowMicros;

  long env[NUM_EMG];

  for (int i = 0; i < NUM_EMG; i++) {
    int raw = analogRead(EMG_PINS[i]);
    int filtered = emgFilter[i].update(raw);
    long sq = (long)filtered * (long)filtered;

    float kEnv = kf[i].update((float)sq);
    if (kEnv < 0) kEnv = 0;

    env[i] = (long)kEnv;
  }

  computeRuntimeFeaturesAndClassify(env);
  moveServosForGesture(currentGesture);

  if (millis() - lastPrintTime > PRINT_INTERVAL_MS) {
    lastPrintTime = millis();
    Serial.print("Gesture: ");
    Serial.print(gestureNames[currentGesture]);
    Serial.print("  ENV: ");
    for (int i = 0; i < NUM_EMG; i++) {
      Serial.print(env[i]);
      Serial.print(" ");
    }
    Serial.println();
  }
}

// ====================================================
//                FEATURE EXTRACTION
// ====================================================
void computeRuntimeFeaturesAndClassify(const long *env) {
  for (int i = 0; i < NUM_EMG; i++) {
    float e = (float)env[i];
    rmsAccum[i] += e * e;
  }
  rmsCount++;

  if (millis() - lastFeatureTime >= CLASS_WINDOW_MS && rmsCount > 0) {
    float rms[NUM_EMG];
    for (int i = 0; i < NUM_EMG; i++) {
      rms[i] = sqrt(rmsAccum[i] / rmsCount);
      rmsAccum[i] = 0;
    }
    rmsCount = 0;
    lastFeatureTime = millis();

    float feat[FEATURE_DIM];
    buildFeatureVector(rms, feat);

    currentGesture = classifyGestureKNN(feat);
  }
}

// ---------------------------------- Build 7-D Feature Vector ----------------------------------
void buildFeatureVector(const float rms[NUM_EMG], float out[FEATURE_DIM]) {
  float B = rms[0];
  float F0 = rms[1];
  float F1 = rms[2];
  float F2 = rms[3];

  float T = B + F0 + F1 + F2 + 1e-6;

  out[0] = B  / T;
  out[1] = F0 / T;
  out[2] = F1 / T;
  out[3] = F2 / T;

  int maxF = 0;
  float m = F0;
  if (F1 > m) { m = F1; maxF = 1; }
  if (F2 > m) { m = F2; maxF = 2; }

  out[4] = (maxF == 0) ? 1 : 0;
  out[5] = (maxF == 1) ? 1 : 0;
  out[6] = (maxF == 2) ? 1 : 0;
}

// ====================================================
//                   k-NN CLASSIFICATION
// ====================================================
int classifyGestureKNN(const float *feat) {
  float bestDist = 1e30;
  int bestG = 0;

  for (int g = 0; g < NUM_GESTURES; g++) {
    for (int s = 0; s < SAMPLES_PER_GESTURE; s++) {

      float dist = 0;
      for (int d = 0; d < FEATURE_DIM; d++) {
        float diff = feat[d] - knnData[g][s][d];
        dist += diff * diff;
      }

      if (dist < bestDist) {
        bestDist = dist;
        bestG = g;
      }
    }
  }
  return bestG;
}

// ====================================================
//                      SERVO CONTROL
// ====================================================
void moveServosForGesture(int g) {
  int targets[NUM_SERVOS] = {0};

  switch (g) {
    case 0: // OpenPalm
      break;

    case 1: // Fist
      for (int i = 0; i < NUM_SERVOS; i++) targets[i] = SERVO_MAX_ANGLE;
      break;

    case 2: // Pinch
      targets[0] = SERVO_MAX_ANGLE; // Thumb
      targets[1] = SERVO_MAX_ANGLE; // Index
      break;

    case 3: // Peace
      targets[1] = SERVO_MAX_ANGLE; // Index
      targets[2] = SERVO_MAX_ANGLE; // Middle
      break;
  }

  for (int i = 0; i < NUM_SERVOS; i++) {
    servoPos[i] = (1.0 - SERVO_SMOOTH)*servoPos[i] + SERVO_SMOOTH*targets[i];
    servos[i].write(servoPos[i]);
  }
}

// ====================================================
//                    CALIBRATION
// ====================================================
void runCalibration() {
  Serial.println("\n--- k-NN Gesture Calibration ---");
  gestureCalibration();
  Serial.println("--- Calibration Finished ---\n");
}

// ---------- Gesture phase ONLY (no relax, no thresholds) ----------
void gestureCalibration() {
  const unsigned long GESTURE_WINDOW_MS = 1500;

  for (int g = 0; g < NUM_GESTURES; g++) {
    Serial.println();
    Serial.print("Prepare gesture: ");
    Serial.println(gestureNames[g]);

    for (int s = 0; s < SAMPLES_PER_GESTURE; s++) {
      Serial.print("  Sample ");
      Serial.print(s+1);
      Serial.print("/");
      Serial.println(SAMPLES_PER_GESTURE);
      Serial.println("  HOLD gesture...");

      delay(1000);

      for (int i = 0; i < NUM_EMG; i++)
        kf[i].init(KALMAN_Q, KALMAN_R);

      float accumSq[NUM_EMG] = {0};
      int count = 0;
      unsigned long start = millis();

      while (millis() - start < GESTURE_WINDOW_MS) {
        for (int i = 0; i < NUM_EMG; i++) {
          int raw = analogRead(EMG_PINS[i]);
          int flt = emgFilter[i].update(raw);
          long sq = (long)flt * (long)flt;
          float kEnv = kf[i].update((float)sq);
          if (kEnv < 0) kEnv = 0;
          accumSq[i] += kEnv * kEnv;
        }
        count++;
      }

      float rms[NUM_EMG];
      for (int i = 0; i < NUM_EMG; i++)
        rms[i] = sqrt(accumSq[i] / count);

      float feat[FEATURE_DIM];
      buildFeatureVector(rms, feat);

      for (int d = 0; d < FEATURE_DIM; d++)
        knnData[g][s][d] = feat[d];

      Serial.print("  ✓ Feature vector: ");
      for (int d = 0; d < FEATURE_DIM; d++) {
        Serial.print(feat[d], 3);
        Serial.print(" ");
      }
      Serial.println();
    }
  }
}
