#include <SoftwareSerial.h>
const int AOUTpin=0;//the AOUT pin of the alcohol sensor goes into analog pin A0 of the arduino
const int DOUTpin=8;//the DOUT pin of the alcohol sensor goes into digital pin D8 of the arduino
//const int ledPin=13;//the anode of the LED connects to digital pin D13 of the arduino

int limit;
float value;
SoftwareSerial mySerial(10, 11);

void setup()
{
  mySerial.begin(9600);  // Setting the baud rate of GSM Module 
  Serial.begin(9600); // Setting the baud rate of Serial Monitor (Arduino)
  pinMode(AOUTpin, INPUT);//sets the pin as an input to the arduino
  pinMode(LED_BUILTIN, OUTPUT);//sets the pin as an output of the arduino
  pinMode(4, INPUT);
  pinMode(9, OUTPUT);
  pinMode(13, OUTPUT);
  delay(100);
}

void loop() 
{
  if (digitalRead(4) == 1)
  {
    value= analogRead(AOUTpin);//reads the analaog value from the alcohol sensor's AOUT pin
    limit= digitalRead(DOUTpin);//reads the digital value from the alcohol sensor's DOUT pin
    Serial.print("Alcohol value: ");
    Serial.println(value);//prints the alcohol value
    Serial.print("Limit: ");
    Serial.println(limit);//prints the limit reached as either LOW or HIGH (above or underneath)
    float v= (value/10) * (5.0/1024.0);
    float mgl= (v*0.67);
    Serial.println(mgl);
    delay(1000);
    if (mgl > 0.08){
      digitalWrite(9, LOW);//if limit has been reached, LED turns on as status indicator
       mySerial.println("AT+CMGF=1");    //Sets the GSM Module in Text Mode
     delay(1000);  // Delay of 1 second
     mySerial.println("AT+CMGS=\"+919427966656\"\r"); // Replace x with mobile number
     delay(8000);
     mySerial.println("Sorry you are drunk you can not drive the car");// The SMS text you want to send
     delay(100);
     mySerial.println((char)26);// ASCII code of CTRL+Z for saying the end of sms to  the module 
    delay(8000);
    break;
    }
    else{
      digitalWrite(9, HIGH);//if threshold not reached, LED remains off
     // Serial.println("Hish");
       mySerial.println("AT+CMGF=1");    //Sets the GSM Module in Text Mode
     delay(1000);  // Delay of 1 second
     mySerial.println("AT+CMGS=\"+919427966656\"\r"); // Replace x with mobile number
     delay(8000);
     mySerial.println("Congrats you can drive the car!!!");// The SMS text you want to send
     delay(100);
     mySerial.println((char)26);// ASCII code of CTRL+Z for saying the end of sms to  the module 
    delay(8000);
    break;
    }
    break;
  }
  else
  {
    digitalWrite(13, HIGH);
    delay(1000);
    digitalWrite(13, LOW);
    delay(1000);
    digitalWrite(13, HIGH);
    delay(1000);
    digitalWrite(13, LOW);
    delay(1000);
    Serial.println(digitalRead(4));
  }
}
