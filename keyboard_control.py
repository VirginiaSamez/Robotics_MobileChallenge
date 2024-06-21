import picar_4wd as fc
import sys
import tty
import termios
import asyncio
import time
import math

power_val = 50
key = 'status'
print("If you want to quit.Please press z")

# Constant
PI = math.pi
FORWARD_SPEED = 0.45 # meters per second
ANGULAR_VELOCITY_LEFT = 2*PI/2.65  # 360 degrees in 3.55 seconds = 1.772 rad/s (counter-clockwise)
ANGULAR_VELOCITY_RIGHT = 2*PI/2.60  # 360 degrees in 3.45 seconds = 1.823 rad/s (clockwise)
current_position = (0, 0)  # (x, y)
current_orientation = PI / 2
start_time = None

def print_position():
    print(f"Position: {current_position}, Orientation: {current_orientation} degrees")
    

# Functions
def update_position(direction, time_elapsed):
    print("Time: ", time_elapsed)
    global current_position, current_orientation

    if direction == 'forward':
        print("current orientation =", current_orientation)
        distance = FORWARD_SPEED * time_elapsed
        rad = current_orientation
        print("distance = ", distance)
        dx = distance * math.cos(rad)
        dy = distance * math.sin(rad)
        print("x =", dx)
        print("y =", dy)
        current_position = (current_position[0] + dx, current_position[1] + dy)
        print_position()

    elif direction == 'backward':
        distance = -FORWARD_SPEED * time_elapsed
        dx = distance * math.cos(current_orientation)
        dy = distance * math.sin(current_orientation)
        current_position = (current_position[0] + dx, current_position[1] + dy)
        print_position()
    
    elif direction == 'turn_left':
        angle_change = ANGULAR_VELOCITY_LEFT * time_elapsed
        current_orientation = (current_orientation + angle_change) % (2*PI)
        print("Left turn finished, ", current_orientation)
        print_position()

    elif direction == 'turn_right':
        angle_change = ANGULAR_VELOCITY_RIGHT * time_elapsed
        current_orientation = (current_orientation - angle_change) % (2*PI)
        print("Right turn finished, ", current_orientation)
        print_position() 


# def return_home():
#     fc.stop()
#     print("Home")


def readchar():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def readkey(getchar_fn=None):
    getchar = getchar_fn or readchar
    c1 = getchar()
    if ord(c1) != 0x1b:
        return c1
    c2 = getchar()
    if ord(c2) != 0x5b:
        return c1
    c3 = getchar()
    return chr(0x10 + ord(c3) - 65)

def back_home():
    if current_position[0] > 0 and current_position[1] > 0:
        alpha = math.acos(current_position[0]/math.sqrt(current_position[0]**2+current_position[1]**2))
    elif current_position[0] < 0 and current_position[1]>0:
        alpha = math.acos(current_position[1]/math.sqrt(current_position[0]**2+current_position[1]**2)) + math.pi/2
    elif current_position[0] < 0 and current_position[1]<0:
        alpha = math.acos(abs(current_position[0])/math.sqrt(current_position[0]**2+current_position[1]**2)) + math.pi
    elif current_position[0] > 0 and current_position[1]<0:
        alpha = math.acos(abs(current_position[1])/math.sqrt(current_position[0]**2+current_position[1]**2)) + 3*math.pi/2
    alpha = (alpha + PI) % (2*PI)
    beta = abs(alpha - current_orientation)
    if beta == 0:
        time_turn = 0
    else:
        time_turn = abs((beta / ANGULAR_VELOCITY_LEFT) /2 + (beta / ANGULAR_VELOCITY_RIGHT) /2)
    time_forward = abs(math.sqrt(current_position[0]**2+current_position[1]**2) / FORWARD_SPEED)
    if current_orientation > alpha:
        fc.turn_right(power_val)
        time.sleep(time_turn)
        fc.stop()
    else:
        fc.turn_left(power_val)
        time.sleep((time_turn * 1.07))
        fc.stop()
    print("turn angle:",(beta * 180/PI)%360)
    print("current orientation", (current_orientation * 180/PI)%360)
    fc.forward(power_val)
    time.sleep((time_forward*1.10))
    fc.stop()
    
    
def Keyborad_control():
    while True:
        global power_val
        global start_time
        key=readkey()
        if key=='6':
            if power_val <=90:
                power_val += 10
                print("power_val:",power_val)
        elif key=='4':
            if power_val >=10:
                power_val -= 10
                print("power_val:",power_val)
        
        if key in {'w', 'a', 's', 'd', 'q'}:
            if (start_time != None):
                print("Key changed! ", key)
                time_elapsed = time.time() - start_time
                update_position(direction, time_elapsed)
                start_time = time.time()
            else:
                start_time = time.time()
            
            if key == 'w':
                fc.forward(power_val)
                direction = 'forward'
            elif key == 'a':
                fc.turn_left(power_val)
                direction = 'turn_left'
            elif key=='s':
                fc.backward(power_val)
                direction = 'backward'
            elif key == 'd':
                fc.turn_right(power_val)
                direction = 'turn_right'
            elif key == 'q':
                back_home()
                current_position = (0,0)
                current_orientation = PI / 2
        else:
            fc.stop()
            print("----- STOP ------")
            time_elapsed = time.time() - start_time
            update_position(direction, time_elapsed)
        if key == 'z':
            print("quit")
            fc.stop()
            break

if __name__ == '__main__':
    Keyborad_control()
