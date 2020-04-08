#include "MPU9250.h"
#include "WiFi.h"
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#define MAX_SIZE 500;
class ArrayList{
  private:
  int index;
  int arraySize;// Size of the array.
  int size; // Number of elements in the array.
  String *data = nullptr;// Pointer to the array.
  public:
  ArrayList() {
    arraySize = MAX_SIZE;
    size = 0;
    index=0;
    data = new String[arraySize];
}
~ArrayList() {
//    delete arraySize;
//    delete size;
    delete []data;
};
void add(String &d) {
    if(index>arraySize-1){
      index=0;
    data[index++] = d;
    }else
    {
      data[index++] = d;
      
    }
    if(size+1< arraySize){
      size++;
    }
}
void clear() {
    size = 0;
    arraySize = MAX_SIZE;
    index=0;
    delete []data;
    data = new String[arraySize];
}
String toString(){
  String ans="";
  for(int i=0; i<=size;i++){
    ans=ans+data[i]+"\n";
  }
  return ans;
}
int getSize(){
  return size;
}
int getIndex(){
  return index;
}
};






//Enter your SSID and PASSWORD
const char* ssid = "TP-LINK_DE9742";
const char* password = "tvroom123";
ArrayList* list=new ArrayList();
MPU9250 IMU(Wire,0x68);

int status;
int i=0;
AsyncWebServer server(80);
const char* PARAM_INPUT_1 = "input1";
const char* PARAM_INPUT_2 = "input2";
const char* PARAM_INPUT_3 = "input3";

// HTML web page to handle 3 input fields (input1, input2, input3)
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE HTML><html><head>
  <title>ESP Input Form</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  </head><body>
  <form action="/get">
    input1: <input type="text" name="input1">
    <input type="submit" value="Submit">
  </form><br>
  <form action="/get">
    input2: <input type="text" name="input2">
    <input type="submit" value="Submit">
  </form><br>
  <form action="/get">
    input3: <input type="text" name="input3">
    <input type="submit" value="Submit">
  </form>
</body></html>)rawliteral";

void notFound(AsyncWebServerRequest *request) {
  request->send(404, "text/plain", "Not found");
}
String handleADC() {
 IMU.readSensor();
  // display the data
  //Serial.print("AccelX: ");
  float ax=IMU.getAccelX_mss();
  float ay=IMU.getAccelY_mss();
  float az=IMU.getAccelZ_mss();
  float gx=IMU.getGyroX_rads();
  float gy=IMU.getGyroY_rads();
  float gz=IMU.getGyroZ_rads();
  float mx=IMU.getMagX_uT();
  float my=IMU.getMagY_uT();
  float mz=IMU.getMagZ_uT();
  float temp=IMU.getTemperature_C();
  float pitch = atan2 (ay ,( sqrt ((ax * ax) + (az * az))));
  float roll = atan2(-ax ,( sqrt((ay * ay) + (az * az))));
  float Yh = (my * cos(roll)) - (mz * sin(roll));
  float Xh = (mx * cos(pitch))+(my * sin(roll)*sin(pitch)) + (mz * cos(roll) * sin(pitch));
  float yaw =  atan2(Yh, Xh);
  String adcValue = String(ax,6)+","+String(ay,6)+","+String(az,6)+","+String(gx,6)+","+String(gy,6)+","+String(gz,6)+","+String(yaw,6)+","+String(pitch,6)+","+String(roll,6)+","+String(temp,6);
  return adcValue;
  }
void setup() {
  // serial to display data
  Serial.begin(115200);
  while(!Serial) {}
  
  // start communication with IMU 
  status = IMU.begin();
  if (status < 0) {
    Serial.println("IMU initialization unsuccessful");
    Serial.println("Check IMU wiring or try cycling power");
    Serial.print("Status: ");
    Serial.println(status);
    while(1) {}
  }
  Serial.println();
  Serial.println("Booting Sketch...");
 
/*
//ESP32 As access point
  WiFi.mode(WIFI_AP); //Access Point mode
  WiFi.softAP(ssid, password);
*/
//ESP32 connects to your wifi -----------------------------------
  WiFi.mode(WIFI_STA); //Connectto your wifi
  WiFi.begin(ssid, password);
 
  Serial.println("Connecting to ");
  Serial.print(ssid);
 
  //Wait for WiFi to connect
  while(WiFi.waitForConnectResult() != WL_CONNECTED){      
      Serial.print(".");
    }
    
  //If connection successful show IP address in serial monitor
  Serial.println("");
  Serial.print("Connected to ");
  Serial.println(ssid);
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());  //IP address assigned to your ESP
//----------------------------------------------------------------
server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
    request->send_P(200, "text/html", index_html);
  });

  // Send a GET request to <ESP_IP>/get?input1=<inputMessage>
  server.on("/get", HTTP_GET, [] (AsyncWebServerRequest *request) {
    String inputMessage;
    String inputParam;
    // GET input1 value on <ESP_IP>/get?input1=<inputMessage>
    if (request->hasParam(PARAM_INPUT_1)) {
      inputMessage = request->getParam(PARAM_INPUT_1)->value();
      inputParam = PARAM_INPUT_1;
    }
    // GET input2 value on <ESP_IP>/get?input2=<inputMessage>
    else if (request->hasParam(PARAM_INPUT_2)) {
      inputMessage = request->getParam(PARAM_INPUT_2)->value();
      inputParam = PARAM_INPUT_2;
    }
    // GET input3 value on <ESP_IP>/get?input3=<inputMessage>
    else if (request->hasParam(PARAM_INPUT_3)) {
      inputMessage = request->getParam(PARAM_INPUT_3)->value();
      inputParam = PARAM_INPUT_3;
    }
    else {
      inputMessage = "No message sent";
      inputParam = "none";
    }
    Serial.println("Data Sent");
    String data=list->toString();
    request->send(200, "text/html",data );
    list->clear();
    Serial.println(inputMessage);
  });
  server.onNotFound(notFound);
  server.begin();
}
void loop() {
  String temp=(handleADC());
  list->add(temp);
  Serial.println(list->getSize());
  Serial.println(list->getIndex());
  delay(200);
} 
