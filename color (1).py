import picar_4wd as fc
import sys
import tty
import termios
import time
import math

import cv2
from picamera2 import Picamera2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


color_dict = {'red':[0,4],'orange':[5,18],'yellow':[22,37],'green':[42,85],'blue':[92,110],'purple':[115,165],'red_2':[165,180]}  #Here is the range of H in the HSV color space represented by the color

kernel_5 = np.ones((5,5),np.uint8) #Define a 5×5 convolution kernel with element values of all 1.


# Initialize plotting
fig, ax = plt.subplots()
positions_x, positions_y = [], []
scat = ax.scatter(positions_x, positions_y)
ax.axis([-10, 10, -10, 10])

def update_plot(frame):
    scat.set_offsets(np.column_stack(([positions_x, positions_y])))
    return scat,

ani = FuncAnimation(fig, update_plot, interval=100, blit=True)

power_val = 50
key = 'status'
print("If you want to quit.Please press z")

# Constant
PI = math.pi
FORWARD_SPEED = 0.35                    # m/s
ANGULAR_VELOCITY_LEFT = 2*PI/2.65       # 360 degrees in 3.55 seconds = 1.772 rad/s (counter-clockwise)
ANGULAR_VELOCITY_RIGHT = 2*PI/2.60      # 360 degrees in 3.45 seconds = 1.823 rad/s (clockwise)
current_position = (0, 0)               # (x, y)
current_orientation = PI / 2
start_time = None
SEARCH_ANGULAR_VELOCITY = 2*PI/5    # the turning angular velocity when the robot is searching for the red object

def color_detect(img,color_name):

    # The blue range will be different under different lighting conditions and can be adjusted flexibly.  H: chroma, S: saturation v: lightness
    resize_img = cv2.resize(img, (160,120), interpolation=cv2.INTER_LINEAR)  # In order to reduce the amount of calculation, the size of the picture is reduced to (160,120)
    hsv = cv2.cvtColor(resize_img, cv2.COLOR_BGR2HSV)              # Convert from BGR to HSV
    color_type = color_name
    
    mask = cv2.inRange(hsv,np.array([min(color_dict[color_type]), 120, 50]), np.array([max(color_dict[color_type]), 255, 255]) )           # inRange()：Make the ones between lower/upper white, and the rest black
    if color_type == 'red':
            mask_2 = cv2.inRange(hsv, (color_dict['red_2'][0],120,50), (color_dict['red_2'][1],255,255)) 
            mask = cv2.bitwise_or(mask, mask_2)

    morphologyEx_img = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_5,iterations=1)              # Perform an open operation on the image 

    # Find the contour in morphologyEx_img, and the contours are arranged according to the area from small to large.
    _tuple = cv2.findContours(morphologyEx_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)      
    # compatible with opencv3.x and openc4.x
    if len(_tuple) == 3:
        _, contours, hierarchy = _tuple
    else:
        contours, hierarchy = _tuple
    
    color_area_num = len(contours) # Count the number of contours

    center_x = 0
    
    if color_area_num > 0: 
        for i in contours:    # Traverse all contours
            x,y,w,h = cv2.boundingRect(i)      # Decompose the contour into the coordinates of the upper left corner and the width and height of the recognition object

            # Draw a rectangle on the image (picture, upper left corner coordinate, lower right corner coordinate, color, line width)
            if w >= 8 and h >= 8: # Because the picture is reduced to a quarter of the original size, if you want to draw a rectangle on the original picture to circle the target, you have to multiply x, y, w, h by 4.
                x = x * 4
                y = y * 4 
                w = w * 4
                h = h * 4
                
                center_x = x + w/2
                print(center_x)

                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)  # Draw a rectangular frame
                cv2.putText(img,color_type,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)# Add character description

    return img,mask,morphologyEx_img, center_x

def print_position():
    print(f"Position: {current_position}, Orientation: {current_orientation} degrees")
    
# Functions
def update_position(direction, time_elapsed):
    print("Time: ", time_elapsed)
    global current_position, current_orientation

    if direction == 'forward':
        distance = FORWARD_SPEED * time_elapsed
        dx = distance * math.cos(current_orientation)
        dy = distance * math.sin(current_orientation)
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
        # print("Left turn finished, ", current_orientation)
        # print_position()

    elif direction == 'turn_right':
        angle_change = ANGULAR_VELOCITY_RIGHT * time_elapsed
        current_orientation = (current_orientation - angle_change) % (2*PI)
        # print("Right turn finished, ", current_orientation)
        # print_position()
    
    positions_x.append(current_position[0])
    positions_y.append(current_position[1])
    print("current orientation 1=", current_orientation*360/(2*PI))


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
    global current_orientation
    global current_position
    
    print("------ Back Home ------")
    print("current orientation 2=", current_orientation*360/(2*PI))
    with Picamera2() as camera:
        print("Starting color detection")

        camera.preview_configuration.main.size = (640, 480)
        camera.preview_configuration.main.format = "RGB888"
        camera.preview_configuration.align()
        camera.configure("preview")
        camera.start()
        
        # Initialize values to track the maximum height of the detected rectangle
        max_height_threshold = 41   # Adjusted on tests, represents the proximity to the object
        camera_started = time.time()
        first_camera_delay = None
        
        while True:
            img = camera.capture_array()  # Capture frame
            img, img_2, img_3, center_x = color_detect(img, 'red')  # Color detection function
            _, _, _, h = cv2.boundingRect(img_3)  # Extracting the height of the contour
            cv2.imshow("video", img)  # Display the processed video
            cv2.imshow("mask", img_2)  # Display mask
            cv2.imshow("morphologyEx_img", img_3)  # Display morphological transformation
            
            # save the startup delay
            if (first_camera_delay == None):
                first_camera_delay = time.time()-camera_started
            
            #print("current orientation1", current_orientation)
            if center_x < 280 or center_x > 320:
                fc.turn_left(power_val * 0.05)
            else:
                fc.stop()
                # calculate the turn time
                turn_time = time.time() - first_camera_delay - camera_started
                print("Turn time", turn_time)
                update_position("forward", first_camera_delay)
                angle_change = (SEARCH_ANGULAR_VELOCITY * turn_time) % (2*PI)
                current_orientation = (current_orientation + angle_change) % (2*PI)
                break

            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # If ESC is pressed, exit loop
                break
        #print("current orientation2", current_orientation)
        
        # save the time before the robot goes forward
        time_forward = time.time()
        
        while True:
            img = camera.capture_array()  # Capture frame
            img, img_2, img_3, center_x = color_detect(img, 'red')  # Color detection function
            _, _, _, h = cv2.boundingRect(img_3)  # Extracting the height of the contour
            # cv2.imshow("video", img)  # Display the processed video
            # cv2.imshow("mask", img_2)  # Display mask
            # cv2.imshow("morphologyEx_img", img_3)  # Display morphological transformation
            
            if h < max_height_threshold:
                fc.forward(power_val)  # Move forward until close enough
            else:
                fc.stop()
                print("--- Object reached ---")
                time_elapsed = time.time() - time_forward
                distance = FORWARD_SPEED * time_elapsed
                dx = distance * math.cos(current_orientation)
                dy = distance * math.sin(current_orientation)
                current_position = (current_position[0] + dx, current_position[1] + dy)
                break

            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # If ESC is pressed, exit loop
                break
            
        fc.forward(power_val)
        time.sleep(0.4)
        fc.stop()
        update_position("forward", 0.4)
        cv2.destroyAllWindows()
        camera.close()
        
    # Once arrived at the object, return home:
    if current_position[0] > 0 and current_position[1] > 0:
        alpha = math.acos(current_position[0]/math.sqrt(current_position[0]**2+current_position[1]**2))
    elif current_position[0] < 0 and current_position[1]>0:
        alpha = math.acos(current_position[1]/math.sqrt(current_position[0]**2+current_position[1]**2)) + math.pi/2
    elif current_position[0] < 0 and current_position[1]<0:
        alpha = math.acos(abs(current_position[0])/math.sqrt(current_position[0]**2+current_position[1]**2)) + math.pi
    elif current_position[0] > 0 and current_position[1]<0:
        alpha = math.acos(abs(current_position[1])/math.sqrt(current_position[0]**2+current_position[1]**2)) + 3*math.pi/2
    
    # Calculate the angle to turn
    alpha = alpha + PI
    beta = abs(alpha - current_orientation)

    # Calculate the time to turn and move forward
    time_turn = abs((beta / ANGULAR_VELOCITY_LEFT) /2 + (beta / ANGULAR_VELOCITY_RIGHT) /2)
    time_forward = abs(math.sqrt(current_position[0]**2+current_position[1]**2) / FORWARD_SPEED)

    #print("Going forward for ", time_forward, " seconds")

    if current_orientation > alpha:
        fc.turn_right(power_val)
        time.sleep(time_turn*1.1)
        fc.stop()
        positions_x.append(current_position[0])
        positions_y.append(current_position[1])
    else:
        fc.turn_left(power_val)
        time.sleep(time_turn*1.1)
        fc.stop()
        positions_x.append(current_position[0])
        positions_y.append(current_position[1])
    # print("turn angle:",beta)
    
    
    # Drive forward
    fc.forward(power_val)
    time.sleep(time_forward)
    fc.stop()
    print("------ Home ------")


    
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
    # plt.show(block=False)
    print("main called")
    Keyborad_control()
   

    

