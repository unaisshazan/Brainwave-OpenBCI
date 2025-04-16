void setup() {
  Serial.begin(9600);
  pinMode(13, OUTPUT); // LED indicator for focus
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == '1') {
      digitalWrite(13, HIGH);  // Focused
    } else if (c == '0') {
      digitalWrite(13, LOW);   // Not focused
    }
  }
}
