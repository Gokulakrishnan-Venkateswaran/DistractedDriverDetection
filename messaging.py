import pywhatkit as kit
import datetime

def send_alert(driver_name="Driver"):
    now = datetime.datetime.now()
    hour = now.hour
    minute = now.minute + 1
    kit.sendwhatmsg("+91xxxxxxxxxx", f"{driver_name}, please focus on driving!", hour, minute)