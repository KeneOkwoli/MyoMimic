/*
   MyoMimic – ESP-NOW RECEIVER (Robotic Hand + LCD)
   -------------------------------------------------------------------------
   Receives gesture commands via ESP-NOW and displays on LCD
   All fingers move simultaneously for natural motion
*/

#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>
#include <ESP32Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <math.h>

// LCD Setup (I2C address 0x27, 16 cols, 2 rows)
LiquidCrystal_I2C lcd(0x27, 16, 2);

Servo thumb;
Servo indexFinger;
Servo middle;
Servo ring;
Servo pinky;

// Current positions
float cur_thumb  = 90;
float cur_index  = 90;
float cur_middle = 90;
float cur_ring   = 90;
float cur_pinky  = 90;

// Gesture tracking
int currentGesture = 0;
int lastGesture = -1;

const char* gestureNames[] = {
  "OpenPalm", "Fist", "Pinch", "Peace"
};

// Statistics
unsigned long totalReceived = 0;
unsigned long lastReceiveTime = 0;

// Function Prototypes
void onDataReceive(const esp_now_recv_info_t *info, const uint8_t *data, int len);
void performGesture(int gesture);
void moveAllFingers(float toThumb, float toIndex, float toMiddle, float toRing, float toPinky, int duration = 1000);
void updateLCD(const String& gestureName, int gestureNum);
void OpenPalm();
void Fist();
void Pinch();
void Peace();

// ====================================================
//                   ESP-NOW CALLBACK
// ====================================================
void onDataReceive(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
  if (len == 1) {
    currentGesture = data[0];
    totalReceived++;
    lastReceiveTime = millis();
    
    // Only perform gesture if it changed
    if (currentGesture != lastGesture && currentGesture < 4) {
      Serial.print("Received: Gesture ");
      Serial.print(currentGesture);
      Serial.print(" (");
      Serial.print(gestureNames[currentGesture]);
      Serial.println(")");
      
      performGesture(currentGesture);
      lastGesture = currentGesture;
    }
  }
}

// ====================================================
//                        SETUP
// ====================================================
void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println("\n╔════════════════════════════════════════╗");
  Serial.println("║  MyoMimic - RECEIVER (Hand + LCD)      ║");
  Serial.println("╚════════════════════════════════════════╝\n");
  
  // Initialize LCD
  Wire.begin(21, 22);  // SDA=21, SCL=22
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("MyoMimic RX");
  lcd.setCursor(0, 1);
  lcd.print("Initializing...");
  delay(1000);

  // Initialize Servos
  thumb.setPeriodHertz(50);
  indexFinger.setPeriodHertz(50);
  middle.setPeriodHertz(50);
  ring.setPeriodHertz(50);
  pinky.setPeriodHertz(50);

  thumb.attach(15, 500, 1900);
  indexFinger.attach(16, 500, 1900);
  middle.attach(17, 500, 1900);
  ring.attach(18, 500, 1900);
  pinky.attach(19, 500, 1900);
  
  // Initialize to OpenPalm position
  Serial.println("Moving to OpenPalm position...");
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Init Position");
  lcd.setCursor(0, 1);
  lcd.print("OpenPalm");
  
  moveAllFingers(110, 150, 150, 140, 150);
  delay(500);

  // Initialize ESP-NOW
  WiFi.mode(WIFI_STA);
  
  Serial.print("Receiver MAC: ");
  Serial.println(WiFi.macAddress());
  
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("ESP-NOW Error!");
    return;
  }
  
  Serial.println("ESP-NOW initialized successfully");
  
  // Register receive callback
  esp_now_register_recv_cb(onDataReceive);
  
  // Display ready message
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Ready!");
  lcd.setCursor(0, 1);
  lcd.print("Waiting...");
  
  Serial.println("\n╔════════════════════════════════════════╗");
  Serial.println("║   Ready to receive gestures!           ║");
  Serial.println("╚════════════════════════════════════════╝\n");
}

// ====================================================
//                         LOOP
// ====================================================
void loop() {
  // ESP-NOW handles everything in callback
  // Just check for timeout to show "waiting" message
  if (millis() - lastReceiveTime > 5000 && totalReceived > 0) {
    // Been more than 5 seconds since last message
    // Could update LCD here if desired
  }
  
  delay(10);
}

// ====================================================
//                  GESTURE EXECUTION
// ====================================================
void performGesture(int gesture) {
  switch (gesture) {
    case 0:
      OpenPalm();
      break;
    case 1:
      Fist();
      break;
    case 2:
      Pinch();
      break;
    case 3:
      Peace();
      break;
    default:
      Serial.println("Unknown gesture");
      break;
  }
}

// ====================================================
//            SIMULTANEOUS FINGER MOVEMENT
// ====================================================
// Move ALL fingers simultaneously using sinusoidal ease in/out
void moveAllFingers(float toThumb, float toIndex, float toMiddle, float toRing, float toPinky, int duration) {
  // Constrain angles
  toThumb  = constrain(toThumb, 0, 180);
  toIndex  = constrain(toIndex, 0, 180);
  toMiddle = constrain(toMiddle, 0, 180);
  toRing   = constrain(toRing, 0, 180);
  toPinky  = constrain(toPinky, 0, 180);
  
  // Starting positions
  float fromThumb  = cur_thumb;
  float fromIndex  = cur_index;
  float fromMiddle = cur_middle;
  float fromRing   = cur_ring;
  float fromPinky  = cur_pinky;
  
  int steps = duration / 20;
  
  for (int i = 0; i <= steps; i++) {
    float t = (float)i / steps;
    float eased = 0.5 * (1 - cos(t * PI)); // sine ease in/out
    
    // Calculate all finger positions simultaneously
    float thumbPos  = fromThumb  + (toThumb  - fromThumb)  * eased;
    float indexPos  = fromIndex  + (toIndex  - fromIndex)  * eased;
    float middlePos = fromMiddle + (toMiddle - fromMiddle) * eased;
    float ringPos   = fromRing   + (toRing   - fromRing)   * eased;
    float pinkyPos  = fromPinky  + (toPinky  - fromPinky)  * eased;
    
    // Write all servos at once
    thumb.write((int)thumbPos);
    indexFinger.write((int)indexPos);
    middle.write((int)middlePos);
    ring.write((int)ringPos);
    pinky.write((int)pinkyPos);
    
    delay(20);
  }
  
  // Update stored positions
  cur_thumb  = toThumb;
  cur_index  = toIndex;
  cur_middle = toMiddle;
  cur_ring   = toRing;
  cur_pinky  = toPinky;
}

// ====================================================
//                     LCD UPDATE
// ====================================================
void updateLCD(const String& gestureName, int gestureNum) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Gesture: ");
  lcd.print(gestureNum);
  lcd.setCursor(0, 1);
  lcd.print(gestureName);
}

// ====================================================
//                  GESTURE DEFINITIONS
// ====================================================
void OpenPalm() {
  Serial.println("→ Executing: OpenPalm");
  updateLCD("OpenPalm", 0);
  
  // Thumb, Index, Middle, Ring, Pinky
  moveAllFingers(110, 150, 150, 140, 150);
  
  Serial.println(" Complete\n");
}

void Fist() {
  Serial.println("→ Executing: Fist");
  updateLCD("Fist", 1);
  
  // Thumb, Index, Middle, Ring, Pinky
  moveAllFingers(60, 90, 30, 70, 90);
  
  Serial.println("Complete\n");
}

void Pinch() {
  Serial.println(" Executing: Pinch");
  updateLCD("Pinch", 2);
  
  // Thumb, Index, Middle, Ring, Pinky
  // Thumb + Index closed, others open
  moveAllFingers(60, 90, 150, 140, 150);
  
  Serial.println(" Complete\n");
}

void Peace() {
  Serial.println("→ Executing: Peace");
  updateLCD("Peace", 3);
  
  // Thumb, Index, Middle, Ring, Pinky
  // Index + Middle open, others closed
  moveAllFingers(60, 150, 150, 70, 90);
  
  Serial.println("Complete\n");
}