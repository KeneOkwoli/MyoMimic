/*
   MyoMimic – ESP-NOW SENDER (5 Forearm Sensors)
   -----------------------------------------------
   CRITICAL: Sensors MUST be on MUSCLE BELLIES (8-12cm above wrist)
   
   IMPROVED: Reduced filtering for better signal separation
   
   SENSORS (MID-FOREARM - where muscles bulge):
        CH0: Dorsal (back)           - GPIO33
        CH1: Radial (thumb side)     - GPIO25
        CH2: Dorsal-Radial (corner)  - GPIO26
        CH3: Ventral (palm side)     - GPIO27
        CH4: Ulnar (pinky side)      - GPIO32

   Gestures:
        0: OpenPalm
        1: Fist
        2: Pinch
        3: Peace
*/

#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include "EMGFilters.h"
#include <math.h>

// ==================== RECEIVER MAC ADDRESS ====================
uint8_t receiverMAC[] = {0xC8, 0x2E, 0x18, 0xF7, 0xA2, 0x94};

#define NUM_EMG 5
#define NUM_GESTURES 4
#define FEATURE_DIM 15
#define SAMPLES_PER_GESTURE 15

int EMG_PINS[NUM_EMG] = {33, 32, 34, 35, 36};

const int SAMPLE_RATE = SAMPLE_FREQ_500HZ;
const int HUM_FREQ    = NOTCH_FREQ_50HZ;

const unsigned long CLASS_WINDOW_MS   = 200;
const unsigned long PRINT_INTERVAL_MS = 500;

// Kalman filter - LESS AGGRESSIVE (more responsive)
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

// IMPROVED: Less aggressive filtering
const float KALMAN_Q = 500.0f;   // Was 100 (increased)
const float KALMAN_R = 2000.0f;  // Was 8000 (decreased)

// Moving Average Filter 
#define MA_WINDOW_SIZE 5
struct MovingAverage {
  float buffer[MA_WINDOW_SIZE];
  int index;
  float sum;
  int count;
  
  void init() {
    index = 0;
    sum = 0;
    count = 0;
    for (int i = 0; i < MA_WINDOW_SIZE; i++) buffer[i] = 0;
  }
  
  float update(float value) {
    sum -= buffer[index];
    buffer[index] = value;
    sum += value;
    index = (index + 1) % MA_WINDOW_SIZE;
    if (count < MA_WINDOW_SIZE) count++;
    return sum / count;
  }
};

// Globals 
EMGFilters emgFilter[NUM_EMG];
Kalman1D   kf[NUM_EMG];
MovingAverage maFilter[NUM_EMG];

int currentGesture = 0;
int lastSentGesture = -1;

unsigned long lastSampleMicros = 0;
const unsigned long SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE;

float rmsAccum[NUM_EMG] = {0};
int   rmsCount = 0;
unsigned long lastFeatureTime = 0;
unsigned long lastPrintTime   = 0;

// Store latest RMS for display
float latestRMS[NUM_EMG] = {0};

// kNN storage
float knnData[NUM_GESTURES][SAMPLES_PER_GESTURE][FEATURE_DIM];

const char* gestureNames[NUM_GESTURES] = {
  "OpenPalm", "Fist", "Pinch", "Peace"
};

#define STABILITY_BUFFER_SIZE 5
int gestureHistory[STABILITY_BUFFER_SIZE];
int historyIndex = 0;

// ESP-NOW status
bool espnowReady = false;
int sendSuccessCount = 0;
int sendFailCount = 0;

// Function Prototypes
void runCalibration();
void gestureCalibration();
void computeRuntimeFeaturesAndClassify(const long *env);
void buildFeatureVector(const float rms[NUM_EMG], float out[FEATURE_DIM]);
int classifyGestureKNN(const float *feat);
int getMostFrequentGesture();
void initESPNow();
void sendGestureESPNow(uint8_t gesture);
void OnDataSent(const wifi_tx_info_t *info, esp_now_send_status_t status);
void runDiagnostic();
void showFilteringStages();

// ====================================================
//                   ESP-NOW CALLBACKS
// ====================================================
void OnDataSent(const wifi_tx_info_t *info, esp_now_send_status_t status) {
  if (status == ESP_NOW_SEND_SUCCESS) {
    sendSuccessCount++;
  } else {
    sendFailCount++;
  }
}

// ====================================================
//                   ESP-NOW SETUP
// ====================================================
void initESPNow() {
  Serial.println("\n--- Initializing ESP-NOW ---");
  
  WiFi.mode(WIFI_STA);
  WiFi.begin();
  
  esp_wifi_set_ps(WIFI_PS_NONE);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);
  delay(100);
  
  Serial.print("Sender MAC: ");
  Serial.println(WiFi.macAddress());

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    espnowReady = false;
    return;
  }

  esp_now_register_send_cb(OnDataSent);

  esp_now_peer_info_t peerInfo;
  memset(&peerInfo, 0, sizeof(peerInfo));
  memcpy(peerInfo.peer_addr, receiverMAC, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    espnowReady = false;
    return;
  }
  
  espnowReady = true;
  Serial.println("ESP-NOW ready\n");
}

void sendGestureESPNow(uint8_t gesture) {
  if (!espnowReady) return;
  esp_now_send(receiverMAC, &gesture, sizeof(gesture));
}

// ====================================================
//                        SETUP
// ====================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n╔════════════════════════════════════════╗");
  Serial.println("║  MyoMimic - 5 FOREARM SENSORS         ║");
  Serial.println("║  IMPROVED: Reduced Filtering           ║");
  Serial.println("╚════════════════════════════════════════╝");
  Serial.println("\n[!] CRITICAL PLACEMENT:");
  Serial.println("    Sensors MUST be 8-12cm ABOVE WRIST");
  Serial.println("    (On muscle bellies, NOT at wrist!)");
  Serial.println("\nCommands:");
  Serial.println("  R: Recalibrate");
  Serial.println("  S: ESP-NOW statistics");
  Serial.println("  D: Sensor diagnostic test");
  Serial.println("  F: Filtering stages diagnostic\n");

  initESPNow();

  for (int i = 0; i < NUM_EMG; i++) {
    emgFilter[i].init(SAMPLE_RATE, HUM_FREQ, true, true, true);
    kf[i].init(KALMAN_Q, KALMAN_R);
    maFilter[i].init();
  }

  for (int i = 0; i < STABILITY_BUFFER_SIZE; i++) {
    gestureHistory[i] = 0;
  }

  runCalibration();
  
  Serial.println("\n╔════════════════════════════════════════╗");
  Serial.println("║      Calibration Complete!             ║");
  Serial.println("╚════════════════════════════════════════╝\n");
}

// ====================================================
//                         LOOP
// ====================================================
void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 'r' || c == 'R') {
      runCalibration();
    } else if (c == 's' || c == 'S') {
      Serial.println("\n--- ESP-NOW Statistics ---");
      Serial.print("Sent: ");
      Serial.println(sendSuccessCount + sendFailCount);
      Serial.print("Success: ");
      Serial.println(sendSuccessCount);
      Serial.print("Failed: ");
      Serial.println(sendFailCount);
      if (sendSuccessCount + sendFailCount > 0) {
        float rate = 100.0 * sendSuccessCount / (sendSuccessCount + sendFailCount);
        Serial.print("Success Rate: ");
        Serial.print(rate, 2);
        Serial.println("%\n");
      }
    } else if (c == 'd' || c == 'D') {
      runDiagnostic();
    } else if (c == 'f' || c == 'F') {
      showFilteringStages();
    }
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
    
    kEnv = maFilter[i].update(kEnv);
    env[i] = (long)kEnv;
  }

  computeRuntimeFeaturesAndClassify(env);

  // Send when gesture changes
  if (currentGesture != lastSentGesture) {
    sendGestureESPNow((uint8_t)currentGesture);
    lastSentGesture = currentGesture;
    
    Serial.print("→ SENT: Gesture ");
    Serial.print(currentGesture);
    Serial.print(" (");
    Serial.print(gestureNames[currentGesture]);
    Serial.println(")");
  }
  
  // CONTINUOUS DISPLAY - Print current gesture + RMS values
  if (millis() - lastPrintTime > PRINT_INTERVAL_MS) {
    lastPrintTime = millis();
    
    Serial.print("Gesture: ");
    Serial.print(currentGesture);
    Serial.print(" (");
    Serial.print(gestureNames[currentGesture]);
    Serial.print(") | RMS: [D:");
    Serial.print((int)latestRMS[0]);
    Serial.print(" R:");
    Serial.print((int)latestRMS[1]);
    Serial.print(" DR:");
    Serial.print((int)latestRMS[2]);
    Serial.print(" V:");
    Serial.print((int)latestRMS[3]);
    Serial.print(" U:");
    Serial.print((int)latestRMS[4]);
    Serial.println("]");
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
      latestRMS[i] = rms[i];  // Store for display
      rmsAccum[i] = 0;
    }
    rmsCount = 0;
    lastFeatureTime = millis();

    float feat[FEATURE_DIM];
    buildFeatureVector(rms, feat);

    int rawGesture = classifyGestureKNN(feat);
    
    gestureHistory[historyIndex] = rawGesture;
    historyIndex = (historyIndex + 1) % STABILITY_BUFFER_SIZE;
    
    currentGesture = getMostFrequentGesture();
  }
}

int getMostFrequentGesture() {
  int counts[NUM_GESTURES] = {0};
  
  for (int i = 0; i < STABILITY_BUFFER_SIZE; i++) {
    counts[gestureHistory[i]]++;
  }
  
  int maxCount = 0;
  int mostFrequent = 0;
  for (int i = 0; i < NUM_GESTURES; i++) {
    if (counts[i] > maxCount) {
      maxCount = counts[i];
      mostFrequent = i;
    }
  }
  
  return mostFrequent;
}

void buildFeatureVector(const float rms[NUM_EMG], float out[FEATURE_DIM]) {
  float D = rms[0];
  float R = rms[1];
  float DR = rms[2];
  float V = rms[3];
  float U = rms[4];

  float total = D + R + DR + V + U + 1e-6;

  out[0] = D / total;
  out[1] = R / total;
  out[2] = DR / total;
  out[3] = V / total;
  out[4] = U / total;

  float dorsalTotal = D + DR;
  float ventralTotal = V + U;
  out[5] = dorsalTotal / (dorsalTotal + ventralTotal + 1e-6);
  out[6] = ventralTotal / (dorsalTotal + ventralTotal + 1e-6);

  float radialTotal = R + DR;
  float ulnarTotal = U;
  out[7] = radialTotal / (radialTotal + ulnarTotal + 1e-6);
  out[8] = ulnarTotal / (radialTotal + ulnarTotal + 1e-6);

  out[9] = D;
  out[10] = R;
  out[11] = DR;
  out[12] = V;
  out[13] = U;
  out[14] = max(max(max(max(D, R), DR), V), U);
}

// ====================================================
//                   k-NN CLASSIFICATION
// ====================================================
int classifyGestureKNN(const float *feat) {
  int votes[NUM_GESTURES] = {0};
  const int K = 3;
  
  struct DistanceLabel {
    float dist;
    int gesture;
  };
  
  DistanceLabel allDistances[NUM_GESTURES * SAMPLES_PER_GESTURE];
  int distIndex = 0;
  
  for (int g = 0; g < NUM_GESTURES; g++) {
    for (int s = 0; s < SAMPLES_PER_GESTURE; s++) {
      float dist = 0;
      for (int d = 0; d < FEATURE_DIM; d++) {
        float diff = feat[d] - knnData[g][s][d];
        dist += diff * diff;
      }
      allDistances[distIndex].dist = dist;
      allDistances[distIndex].gesture = g;
      distIndex++;
    }
  }
  
  for (int i = 0; i < K; i++) {
    int minIdx = i;
    for (int j = i + 1; j < distIndex; j++) {
      if (allDistances[j].dist < allDistances[minIdx].dist) {
        minIdx = j;
      }
    }
    DistanceLabel temp = allDistances[i];
    allDistances[i] = allDistances[minIdx];
    allDistances[minIdx] = temp;
    
    votes[allDistances[i].gesture]++;
  }
  
  int maxVotes = 0;
  int bestGesture = 0;
  for (int g = 0; g < NUM_GESTURES; g++) {
    if (votes[g] > maxVotes) {
      maxVotes = votes[g];
      bestGesture = g;
    }
  }
  
  return bestGesture;
}

// ====================================================
//            FILTERING STAGES DIAGNOSTIC
// ====================================================
void showFilteringStages() {
  Serial.println("\n╔════════════════════════════════════════╗");
  Serial.println("║   FILTERING STAGES DIAGNOSTIC          ║");
  Serial.println("╚════════════════════════════════════════╝");
  Serial.println("\nThis shows signal at each processing stage");
  Serial.println("RELAX your arm completely...\n");
  
  delay(2000);
  
  Serial.println("=== RELAXED (Baseline) ===\n");
  for (int test = 0; test < 5; test++) {
    for (int i = 0; i < NUM_EMG; i++) {
      int raw = analogRead(EMG_PINS[i]);
      int filtered = emgFilter[i].update(raw);
      long sq = (long)filtered * (long)filtered;
      float afterKalman = kf[i].update((float)sq);
      if (afterKalman < 0) afterKalman = 0;
      float afterMA = maFilter[i].update(afterKalman);
      float rms = sqrt(afterMA);
      
      Serial.print("CH");
      Serial.print(i);
      Serial.print(": Raw=");
      Serial.print(raw);
      Serial.print(" → Filt=");
      Serial.print(filtered);
      Serial.print(" → Sq²=");
      Serial.print(sq);
      Serial.print(" → Kalman=");
      Serial.print((int)afterKalman);
      Serial.print(" → MA=");
      Serial.print((int)afterMA);
      Serial.print(" → RMS=");
      Serial.println((int)rms);
    }
    Serial.println();
    delay(500);
  }
  
  Serial.println("\n>>> NOW MAKE A TIGHT FIST! <<<");
  Serial.println(">>> SQUEEZE AS HARD AS YOU CAN! <<<\n");
  delay(3000);
  
  Serial.println("=== FIST (Active) ===\n");
  for (int test = 0; test < 5; test++) {
    for (int i = 0; i < NUM_EMG; i++) {
      int raw = analogRead(EMG_PINS[i]);
      int filtered = emgFilter[i].update(raw);
      long sq = (long)filtered * (long)filtered;
      float afterKalman = kf[i].update((float)sq);
      if (afterKalman < 0) afterKalman = 0;
      float afterMA = maFilter[i].update(afterKalman);
      float rms = sqrt(afterMA);
      
      Serial.print("CH");
      Serial.print(i);
      Serial.print(": Sq²=");
      Serial.print(sq);
      Serial.print(" → Kalman=");
      Serial.print((int)afterKalman);
      Serial.print(" → MA=");
      Serial.print((int)afterMA);
      Serial.print(" → RMS=");
      Serial.println((int)rms);
    }
    Serial.println();
    delay(500);
  }
  
  Serial.println("\n╔════════════════════════════════════════╗");
  Serial.println("║         DIAGNOSTIC COMPLETE            ║");
  Serial.println("╚════════════════════════════════════════╝");
  Serial.println("\nLOOK FOR:");
  Serial.println(" 'Sq²' should be MUCH higher during fist");
  Serial.println("  (100-10000 relaxed → 10000-100000 fist)");
  Serial.println(" 'Kalman' should track Sq² changes");
  Serial.println(" 'MA' should smooth slightly");
  Serial.println(" 'RMS' is final value (sqrt of MA)");
  Serial.println("\nPROBLEMS:");
  Serial.println(" If Sq² doesn't change → sensor placement");
  Serial.println(" If Kalman barely moves → over-filtering");
  Serial.println(" If all channels identical → bad contacts\n");
}

// ====================================================
//               SENSOR DIAGNOSTIC
// ====================================================
void runDiagnostic() {
  Serial.println("\n╔════════════════════════════════════════╗");
  Serial.println("║        SENSOR DIAGNOSTIC TEST          ║");
  Serial.println("╚════════════════════════════════════════╝");
  Serial.println("\n[1] RELAX your arm completely...\n");
  
  delay(2000);
  
  Serial.println("Reading baseline (relaxed):");
  for (int i = 0; i < 10; i++) {
    Serial.print("  ");
    for (int ch = 0; ch < NUM_EMG; ch++) {
      int raw = analogRead(EMG_PINS[ch]);
      Serial.print("CH");
      Serial.print(ch);
      Serial.print("=");
      Serial.print(raw);
      Serial.print(" ");
    }
    Serial.println();
    delay(200);
  }
  
  Serial.println("\n[2] NOW MAKE A TIGHT FIST!");
  Serial.println("    (Squeeze as HARD as you can!)\n");
  
  delay(3000);
  
  Serial.println("Reading during FIST:");
  for (int i = 0; i < 10; i++) {
    Serial.print("  ");
    for (int ch = 0; ch < NUM_EMG; ch++) {
      int raw = analogRead(EMG_PINS[ch]);
      Serial.print("CH");
      Serial.print(ch);
      Serial.print("=");
      Serial.print(raw);
      Serial.print(" ");
    }
    Serial.println();
    delay(200);
  }
  
  Serial.println("\n╔════════════════════════════════════════╗");
  Serial.println("║         DIAGNOSTIC COMPLETE            ║");
  Serial.println("╚════════════════════════════════════════╝");
  Serial.println("\nLOOK FOR:");
  Serial.println(" Baseline should be ~2048");
  Serial.println(" Values should CHANGE 50-200 during fist");
  Serial.println(" If stuck at same value → bad contact");
  Serial.println(" If all ~2048 → sensors NOT on muscles!");
  Serial.println(" Move sensors to MID-FOREARM (8-12cm");
  Serial.println("   above wrist, where muscles bulge)\n");
}

// ====================================================
//                    CALIBRATION
// ====================================================
void runCalibration() {
  Serial.println("\n╔════════════════════════════════════════╗");
  Serial.println("║          k-NN Calibration              ║");
  Serial.println("║     Improved Filtering (Less Aggressive) ║");
  Serial.println("╚════════════════════════════════════════╝");
  gestureCalibration();
  Serial.println("\n╔════════════════════════════════════════╗");
  Serial.println("║      Calibration Complete!             ║");
  Serial.println("╚════════════════════════════════════════╝\n");
}

void gestureCalibration() {
  const unsigned long GESTURE_WINDOW_MS = 2000;
  const unsigned long WARMUP_MS = 2000;  // Warm-up period

  for (int g = 0; g < NUM_GESTURES; g++) {
    Serial.println();
    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    Serial.print(gestureNames[g]);
    Serial.print(": Gesture ");
    Serial.println(g);
    
    if (g == 0) Serial.println("   → Complete relaxation");
    else if (g == 1) Serial.println("   → MAXIMUM FIST - CRUSH IT!");
    else if (g == 2) Serial.println("   → HARD PINCH - MAXIMUM FORCE!");
    else if (g == 3) Serial.println("   → PEACE - FULL EXTENSION!");
    
    Serial.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    for (int s = 0; s < SAMPLES_PER_GESTURE; s++) {
      Serial.println();
      Serial.print("  Sample ");
      Serial.print(s + 1);
      Serial.print("/");
      Serial.println(SAMPLES_PER_GESTURE);
      Serial.print("  Prepare in ");
      
      for (int countdown = 3; countdown > 0; countdown--) {
        Serial.print(countdown);
        Serial.print("... ");
        delay(1000);
      }
      Serial.println("HOLD!");

      // Only reset filters on very first sample of very first gesture
      if (g == 0 && s == 0) {
        Serial.println("  [Initializing filters...]");
        for (int i = 0; i < NUM_EMG; i++) {
          kf[i].init(KALMAN_Q, KALMAN_R);
          maFilter[i].init();
        }
      }
      
      // Warm-up period - let filters settle to gesture level
      Serial.print("  Warming up filters");
      unsigned long warmupStart = millis();
      while (millis() - warmupStart < WARMUP_MS) {
        for (int i = 0; i < NUM_EMG; i++) {
          int raw = analogRead(EMG_PINS[i]);
          int flt = emgFilter[i].update(raw);
          long sq = (long)flt * (long)flt;
          float kEnv = kf[i].update((float)sq);
          if (kEnv < 0) kEnv = 0;
          kEnv = maFilter[i].update(kEnv);
        }
        
        // Print progress dots every 500ms
        if ((millis() - warmupStart) % 500 == 0) {
          Serial.print(".");
        }
        delayMicroseconds(100);
      }
      Serial.println(" Ready!");

      // Capture the actual sample with settled filters
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
          kEnv = maFilter[i].update(kEnv);
          accumSq[i] += kEnv * kEnv;  // FIXED: Square kEnv to match runtime calculation
        }
        count++;
        delayMicroseconds(100);
      }

      // Calculate RMS (now matches runtime calculation)
      float rms[NUM_EMG];
      for (int i = 0; i < NUM_EMG; i++)
        rms[i] = sqrt(accumSq[i] / count);

      // Build feature vector and store in kNN training data
      float feat[FEATURE_DIM];
      buildFeatureVector(rms, feat);

      for (int d = 0; d < FEATURE_DIM; d++)
        knnData[g][s][d] = feat[d];

      // Display RMS values (should now be 3-4 digits)
      Serial.print("   RMS: [D:");
      Serial.print((int)rms[0]);
      Serial.print(" R:");
      Serial.print((int)rms[1]);
      Serial.print(" DR:");
      Serial.print((int)rms[2]);
      Serial.print(" V:");
      Serial.print((int)rms[3]);
      Serial.print(" U:");
      Serial.print((int)rms[4]);
      Serial.println("]");
      
      delay(500);
    }
    
    Serial.println();
    Serial.print(" Completed ");
    Serial.print(gestureNames[g]);
    Serial.println();
    delay(1000);
  }
  
  Serial.println("\n--- Training Complete ---");
  Serial.print("Total samples: ");
  Serial.println(NUM_GESTURES * SAMPLES_PER_GESTURE);
}
