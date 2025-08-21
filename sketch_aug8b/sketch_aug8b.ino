/* Combined PPG (MAX30101 via SparkFun Bio Hub) + MPU6050 exporter
   - Prints a header line once on start and whenever a new client connects
   - Then prints 1 CSV line per loop with:
     arduino_ms,PPG_HR,PPG_O2,PPG_Conf,PPG_Status,
     MPU_AccelX,MPU_AccelY,MPU_AccelZ,MPU_GyroX,MPU_GyroY,MPU_GyroZ,MPU_Temp,
     MPU_VelX,MPU_VelY,MPU_VelZ
   - Loop delay ~10ms (100 Hz)
   - Streams CSV wirelessly over Wi-Fi (TCP port 8080)
*/

#include <Wire.h>
#include <SparkFun_Bio_Sensor_Hub_Library.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <WiFi.h>

// ==== Wi-Fi Settings ====
const char* ssid     = "Zyxel_DCBB";     
const char* password = "CPRQQXPTP3";     
WiFiServer server(8080);
WiFiClient client;

// ==== Pin Mapping ====
#define PPG_SDA 21
#define PPG_SCL 22
#define PPG_RESET_PIN 4
#define PPG_MFIO_PIN 13

// ==== Sensors ====
Adafruit_MPU6050 mpu;
SparkFun_Bio_Sensor_Hub bioHub(PPG_RESET_PIN, PPG_MFIO_PIN);
bioData body;

// ==== Motion vars ====
float lastAccelX = 0, lastAccelY = 0, lastAccelZ = 0;
float velocityX = 0, velocityY = 0, velocityZ = 0;
unsigned long lastMotionTime = 0;

void print_header(WiFiClient &c) {
  const char* header =
    "arduino_ms,PPG_HR,PPG_O2,PPG_Conf,PPG_Status," 
    "MPU_AccelX,MPU_AccelY,MPU_AccelZ,MPU_GyroX,MPU_GyroY,MPU_GyroZ,MPU_Temp," 
    "MPU_VelX,MPU_VelY,MPU_VelZ";
  Serial.println(header);
  if (c && c.connected()) c.println(header);
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin(PPG_SDA, PPG_SCL);

  // ==== Wi-Fi Init ====
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected!");
  Serial.print("ESP32 IP address: ");
  Serial.println(WiFi.localIP());
  server.begin();

  // ==== MPU6050 Init ====
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) delay(10);
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_16_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // ==== PPG Init ====
  int result = bioHub.begin();
  if (result == 0) Serial.println("PPG sensor started!");
  else Serial.println("PPG sensor not found!");
  bioHub.configBpm(MODE_ONE);
  delay(2000);

  lastMotionTime = millis();
}

void loop() {
  unsigned long now = millis();

  // ==== Accept client ====
  if (!client || !client.connected()) {
    WiFiClient newClient = server.available();
    if (newClient) {
      Serial.println("New TCP client connected");
      client = newClient;
      print_header(client);  // send CSV header to new client
    }
  }

  // ==== PPG ====
  body = bioHub.readBpm();
  float ppg_hr = body.heartRate;
  float ppg_o2 = body.oxygen;
  int ppg_conf = body.confidence;
  int ppg_status = body.status;

  // ==== MPU6050 ====
  sensors_event_t a, g, tempEvent;
  mpu.getEvent(&a, &g, &tempEvent);

  // Velocity integration
  float dt = (now - lastMotionTime) / 1000.0;
  velocityX += (a.acceleration.x + lastAccelX) / 2.0 * dt;
  velocityY += (a.acceleration.y + lastAccelY) / 2.0 * dt;
  velocityZ += (a.acceleration.z + lastAccelZ) / 2.0 * dt;
  lastAccelX = a.acceleration.x;
  lastAccelY = a.acceleration.y;
  lastAccelZ = a.acceleration.z;
  lastMotionTime = now;

  // ==== Format CSV line ====
  String line = String(now) + "," +
                String(ppg_hr) + "," + String(ppg_o2) + "," + String(ppg_conf) + "," + String(ppg_status) + "," +
                String(a.acceleration.x) + "," + String(a.acceleration.y) + "," + String(a.acceleration.z) + "," +
                String(g.gyro.x) + "," + String(g.gyro.y) + "," + String(g.gyro.z) + "," +
                String(tempEvent.temperature) + "," +
                String(velocityX) + "," + String(velocityY) + "," + String(velocityZ);

  // ==== Send ====
  Serial.println(line);
  if (client && client.connected()) {
    client.println(line);
  }

  delay(10); // ~100 Hz
}
