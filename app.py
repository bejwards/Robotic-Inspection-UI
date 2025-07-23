from flask import Flask, render_template, send_file, request, Response, jsonify
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
from functools import wraps
import sqlite3
import pandas as pd
import os
import shutil 
from datetime import datetime
from PIL import Image
import time

app = Flask(__name__)

class sensordata: 
    def __init__(self): 
        #creating file path
        dbfile = 'calibration1.db'
        con = sqlite3.connect(dbfile)   #creating a SQL connection to our SQlite database
        cur = con.cursor()  # creating cursor

        # reading all table names
        table_list = [a for a in cur.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        sensor_df = pd.read_sql_query("SELECT * from data", con)

        con.close() #closed connection to database. All data is stored/accessed from sensor_df (dataframe) 

        R = (sensor_df.iloc[0:1896,15:31].to_numpy(dtype=float))    #unsure if needed, however converts to numpy matrix (array)  
        I = (sensor_df.iloc[0:1896,1:15].to_numpy(dtype=float))  

        B_R = R[0,:]
        B_I = I[0,:]

        self.R_P = (R-B_R)               
        self.R_P = np.divide(self.R_P,R)

        self.I_P = (I-B_I)
        self.I_P = np.divide(self.I_P,I)

        self.R_data = self.R_P[0,:]
        self.I_data = self.I_P[0,:]
        self.counter = 0
    
    def increment_data(self): 
        self.R_data = self.R_P[self.counter,:]
        self.I_data = self.I_P[self.counter,:]

        if (self.counter ==1895):
                self.counter = 0

        self.counter = self.counter+1


        #print(self.R_data)

class Superclass: 

    def __init__(self):
    #initialise sensor data class: 
        self.sensor_data = sensordata()
    
    #COIL STUFF
        self.x_sep = 10           #distance from robot centre to centre of coils            
        self.coil_sep = 3      #distance between centres of adjacent coils (2*channel width) 
        self.coil_row_sep = np.sqrt((self.coil_sep**2)*3/4)   #vertical sepation between 2 rows of coils

        #channel "vertices" (boundaries) coordinates at robot 0,0,theta=0
        self.vert_d = np.array([[self.x_sep, self.x_sep, self.x_sep, self.x_sep, self.x_sep, self.x_sep, self.x_sep, self.x_sep, self.x_sep, self.x_sep, self.x_sep, self.x_sep, self.x_sep, self.x_sep, self.x_sep],   #vertices x values
                                 [self.coil_sep*15/4,self.coil_sep*13/4,self.coil_sep*11/4,self.coil_sep*9/4,self.coil_sep*7/4,self.coil_sep*5/4,self.coil_sep*3/4,self.coil_sep*1/4,-self.coil_sep*1/4,-self.coil_sep*3/4,-self.coil_sep*5/4,-self.coil_sep*7/4,-self.coil_sep*9/4,-self.coil_sep*11/4,-self.coil_sep*13/4]]).T #vertices y values
    
        self.R = np.array([[np.cos(0), np.sin(0)],      #initialise rotation matrix
                           [-np.sin(0), np.cos(0)]])

        self.robot_coord = np.zeros((2))    #initialse robot coordinates

        self.vert_coord = self.vert_d       #initalise vertex coordinates
        self.vert_coord_prev = self.vert_d

        self.triangle_coordinates = np.zeros((28,3,2))  #initialise triangle array

    #IMAGE STUFF 
        self.sample_width = 800
        self.sample_height = 800

    #RASTERIZATION STUFF
        self.width = 2500
        self.height = 2500
        self.bbmin = np.zeros((28,2))
        self.bbmax = np.zeros((28,2))

        self.bb_x_index = np.zeros((28,2))
        self.bb_y_index = np.zeros((28,2))

        self.pixel_array = np.empty((self.width,self.height,3))
        self.pixel_array.fill(np.nan)

    #coils functions
    def calc_vertices(self,x,y,theta): 
        self.vert_coord_prev = self.vert_coord  #store previous vertex coordinates

        self.R = [[np.cos(theta), np.sin(theta)],      #initialise rotation matrix
                 [-np.sin(theta), np.cos(theta)]]

        self.robot_coord[0] = x #update robot coordinates
        self.robot_coord[1] = y

        self.vert_coord = self.robot_coord + self.vert_d @ self.R   #calculate coordinates

    #ideally want to get rid of with some clever indexing. 
    def create_triangles(self):
        #change to for i in range 14 maybe
        for i in range (28): 
            if i % 2 == 0:  #even numbers
                #vertex0
                self.triangle_coordinates[i][0][0] = self.vert_coord_prev[int(i/2)][0]
                self.triangle_coordinates[i][0][1] = self.vert_coord_prev[int(i/2)][1]
                #vertex1
                self.triangle_coordinates[i][1][0] = self.vert_coord[int(i/2)][0]
                self.triangle_coordinates[i][1][1] = self.vert_coord[int(i/2)][1]
                #vertex2
                self.triangle_coordinates[i][2][0] = self.vert_coord_prev[int(i/2+1)][0]
                self.triangle_coordinates[i][2][1] = self.vert_coord_prev[int(i/2+1)][1]
 
            else:   #odd numbers
                #vertex0
                self.triangle_coordinates[i][0][0] = self.vert_coord[int((i-1)/2)][0]
                self.triangle_coordinates[i][0][1] = self.vert_coord[int((i-1)/2)][1]
                #vertex1
                self.triangle_coordinates[i][1][0] = self.vert_coord[int(((i-1)/2)+1)][0]
                self.triangle_coordinates[i][1][1] = self.vert_coord[int(((i-1)/2)+1)][1]
                #vertex2
                self.triangle_coordinates[i][2][0] = self.vert_coord_prev[int(((i-1)/2)+1)][0]
                self.triangle_coordinates[i][2][1] = self.vert_coord_prev[int(((i-1)/2)+1)][1]

        #not happy with this function overall. needs rewritten
        for i in range(28): 
            for j in range(3):
                self.triangle_coordinates[i][j] = self.cartNDCconv(self.triangle_coordinates[i][j][0],self.triangle_coordinates[i][j][1])

    #image functions    
    def find_inital_origin(self):
        #x,y coordinates for bottom channel edge. (0,0) 
        self.origin = self.vert_coord[14]

    def cartNDCconv(self, x_abs, y_abs):
        l = 0 
        r = self.sample_width
        b = 0 
        t = self.sample_height 

        #calculate constants earlier??
        self.pNDC = np.zeros(2)
        #creates new origin at location of far right coil at initialisation (initialise before sampling)
        x = x_abs - self.origin[0]
        y = y_abs - self.origin[1]
        #converts to between 0 and 1 for x and y. 
        #self.pNDC[0] = x /(r-l) - l/(r-l)
        #self.pNDC[1] = y/(t-b) - t/(t-b)
        self.pNDC[0] = x/r 
        self.pNDC[1] = y/t 

        return self.pNDC

    #////////////////////////////////////////////////////////////////////////////
    #rasterization functions 
    #rewriting bounding_box for ndc space
    def bounding_box(self):

        for j in range(28):
            self.bbmin[j] = [np.inf,np.inf]
            self.bbmax[j] = [-np.inf,-np.inf]

            for i in range(3):
                #bbmin x and y
                if self.triangle_coordinates[j][i][0] < self.bbmin[j][0]: 
                    self.bbmin[j][0] = self.triangle_coordinates[j][i][0]
                if self.triangle_coordinates[j][i][1] < self.bbmin[j][1]: 
                    self.bbmin[j][1] = self.triangle_coordinates[j][i][1]
                #bbmax x and y
                if self.triangle_coordinates[j][i][0] > self.bbmax[j][0]: 
                    self.bbmax[j][0] = self.triangle_coordinates[j][i][0]
                if self.triangle_coordinates[j][i][1] > self.bbmax[j][1]: 
                    self.bbmax[j][1] = self.triangle_coordinates[j][i][1]

            #conditionals for if it goes off edge of sample.
            if self.bbmin[j][0] < 0: 
                self.bbmin[j][0] = 0
            if self.bbmax[j][0] < 0:
                self.bbmax[j][0] = 0

            if self.bbmin[j][0] > 1: 
                self.bbmin[j][0] = 1
            if self.bbmax[j][0] > 1:
                self.bbmax[j][0] = 1

            if self.bbmin[j][1] < 0: 
                self.bbmin[j][1] = 0
            if self.bbmax[j][1] < 0:
                self.bbmax[j][1] = 0

            if self.bbmin[j][1] > 1: 
                self.bbmin[j][1] = 1
            if self.bbmax[j][1] > 1:
                self.bbmax[j][1] = 1

            self.bbmin[j][0] = self.bbmin[j][0]*self.width
            self.bbmax[j][0] = self.bbmax[j][0]*self.width

            self.bbmin[j][1] = self.bbmin[j][1]*self.height
            self.bbmax[j][1] = self.bbmax[j][1]*self.height

            #print("1 self.bbmin: ",self.bbmin)
            #print("1 self.bbmax: ",self.bbmax)

            #need to adjust this part. 
            self.bbmin[j] = np.floor(self.bbmin[j])   #bbmin round down to nearest whole ints 
            self.bbmax[j] = np.ceil(self.bbmax[j])    #bbmax round up to nearest whole ints

            #print("2 self.bbmin: ",self.bbmin)
            #print("2 self.bbmax: ",self.bbmax)

            #convert to index. 
            self.bb_x_index[j] = [int(self.bbmin[j][0]), int(self.bbmax[j][0])]
            self.bb_y_index[j] = [int(self.height - self.bbmax[j][1]), int(self.height - self.bbmin[j][1])]

            #convert to index. 
            self.bb_x_index[j] = [int(self.bbmin[j][0]), int(self.bbmax[j][0])]
            self.bb_y_index[j] = [int(self.bbmin[j][1]), int(self.bbmax[j][1])]

    def edge_func(self): 

        #work out how to access triangle (loop)
        #calculate bounding box
        self.bounding_box()
        
        for i in range(28):

            xv0 = self.triangle_coordinates[i][0][0]
            yv0 = self.triangle_coordinates[i][0][1]

            xv1 = self.triangle_coordinates[i][1][0]
            yv1 = self.triangle_coordinates[i][1][1]

            xv2 = self.triangle_coordinates[i][2][0]
            yv2 = self.triangle_coordinates[i][2][1]

            #can probably iterate through original array in this loop instead of using triangle_coordinates
            for y_index in range(int(self.bb_y_index[i][0]),int(self.bb_y_index[i][1])): 
                for x_index in range(int(self.bb_x_index[i][0]),int(self.bb_x_index[i][1])): 

                    #NDC FOR PIXELS based on x and y. DOUBLE CHECK
                    xP = (x_index + 0.5)/self.width
                    yP = (y_index + 0.5)/self.height

                    edge1 = (xP - xv0)*(yv1 - yv0) - (yP - yv0)*(xv1 - xv0)
                    edge2 = (xP - xv1)*(yv2 - yv1) - (yP - yv1)*(xv2 - xv1)
                    edge3 = (xP - xv2)*(yv0 - yv2) - (yP - yv2)*(xv0 - xv2)

                    if (edge1 > 0) and (edge2>0) and (edge3>0):
                        #print("x_index, y_index: ",x_index,y_index)
                        #convert from cartesian centred NDC to from top image
                        self.pixel_array[(self.height-1)-y_index][x_index][0] = self.sensor_data.R_data[int(np.floor(i/2))]
                        self.pixel_array[(self.height-1)-y_index][x_index][1] = self.sensor_data.I_data[int(np.floor(i/2))]
                        #flag to say its been written to
                        self.pixel_array[(self.height-1)-y_index][x_index][2] = int(np.floor(i/2)) #zero based indexing of channel number for flag bits
                    elif (edge1 == 0) or (edge2 == 0) or (edge3 == 0): 
                        #DO TOP LEFT RULE HERE
                        overlaps = False

                        line1 = [xv1-xv0,yv1-yv0]
                        line2 = [xv2-xv1,yv2-yv1]
                        line3 = [xv0-xv2,yv0-yv2]
                        
                        if (edge1 == 0): 
                            overlaps = ((line1[1] == 0) and (line1[0] > 0)) or (line1[1] > 0)
                           
                        elif (edge2 == 0): 
                            overlaps = ((line2[1] == 0) and (line2[0] > 0)) or (line2[1] > 0)

                        elif (edge3 == 0): 
                            overlaps = ((line3[1] == 0) and (line3[0] > 0)) or (line3[1] > 0)

                        if(overlaps):
                            self.pixel_array[(self.height-1)-y_index][x_index][0] = self.sensor_data.R_data[int(np.floor(i/2))]
                            self.pixel_array[(self.height-1)-y_index][x_index][1] = self.sensor_data.I_data[int(np.floor(i/2))]
                            #flag to say its been written to
                            self.pixel_array[(self.height-1)-y_index][x_index][2] = int(np.floor(i/2)) #zero based indexing of channel number for flag bits
                    else: 
                        pass
                        #self.pixel_array[x_index][y_index] = 0



                    

    def process(self,x,y,theta):
        
        self.calc_vertices(x,y,theta)
        self.create_triangles()     #create_triangles now includes pNDC conversion. 
        self.edge_func()

    def plot(self): 
        #real plot
        #plt.imshow(self.pixel_array[:,:,0],cmap = 'jet')
        #plt.show()
        ##imag plot 
        #plt.imshow(self.pixel_array[:,:,1],cmap = 'jet')
        #plt.show()

        #binary image of whats been written to
        #plt.imshow(self.pixel_array[:,:,2])
        #plt.show()
        pass

    def output_array(self): 
        return self.pixel_array

class plotter: 

    def __init__(self): 

        #self.image_array = np.zeros((2500,2500,4),dtype=np.uint8)

        self.init_delay = 0         #delay for image counter
        self.inverted_colourmap = False
        self.image_number = 0  #string for sending image to be displayed to Flask script
        self.real_selection = True   #true represents real numbers
        self.p = 150      #sigmoid function variables
        self.k = 0      #between 0 and 2
        self.loop_indicator = 1
        self.colourmap = 'jet'

        #could have intermediary stage where it is an array of normalised coefficients with a flag, before colour mapping. then can perform thresholding on the normalised array to only show scratches. 

    #recieves new raw data array input. potentially update image rather than plot whole new image. 
    def newinput(self,raw_data): 

        #look at this again
        self.raw_data_array = raw_data
        #self.image_array[:,:,3] = self.raw_data_array[:,:,2]*255

        #apply colormap and normalisation
        
    def raw_data_ops(self,userinput,usercolormap):
        #need user input to select real, imaginary averaged and normalisation
        
        if (self.real_selection):           #real numbers
            self.augmented_data = self.raw_data_array[:,:,0]
        else:                                #imaginary numbers
            self.augmented_data = self.raw_data_array[:,:,1]

        self.colourmaps(usercolormap)

        pass

    def colourmaps(self, imgNo):

        #figure out whats wrong with this. 
        normval = mpl.colors.Normalize(vmin=np.nanmin(self.augmented_data), vmax=np.nanmax(self.augmented_data))
        #need to change norm probably
        
        sm = ScalarMappable(norm = normval,cmap=self.colourmap)

        #colored_data = (sm.to_rgba(self.augmented_data)).astype(np.uint8)
        colored_data = (sm.to_rgba(self.augmented_data)[:, :, :4] * 255 ).astype(np.uint8)

        #function for scaling transparency
        min_val = np.nanmin(self.augmented_data)
        max_val = np.nanmax(self.augmented_data)

        normalized_arr = 1 - (self.augmented_data - min_val) / (max_val - min_val)


        #sigmoid function NEED TO ADD OPTIONS WHETHER TO USE SIGMOID FUNCTION OR NOT 
        normalized_arr = 1/(1+np.exp((-(self.p*(normalized_arr-self.k*0.5)))))
        #converts nan values to 0 (makes transparent)
        normalized_arr[np.isnan(normalized_arr)] = 0

        colored_data[:,:,3] = (normalized_arr*255).astype(np.uint8)
        
        img = Image.fromarray(colored_data)

        if (self.image_number == 0 or self.image_number == 2):
            img.save('image/live1.png') 
            self.image_number = 1       
        else:
            img.save(f"image/live{self.image_number + 1}.png")
            self.image_number = 2
      

        #RdYlGn is proper name (similar to eddyfi colors)
        #plotly
        #CREATE DEFAULT PRESETS 
 
    def regenerate(self):
        normval = mpl.colors.Normalize(vmin=np.nanmin(self.augmented_data), vmax=np.nanmax(self.augmented_data))
        #need to change norm probably

        sm = ScalarMappable(norm = normval,cmap=self.colourmap)


        #colored_data = (sm.to_rgba(self.augmented_data)).astype(np.uint8)
        colored_data = (sm.to_rgba(self.augmented_data)[:, :, :4] * 255 ).astype(np.uint8)

        #function for scaling transparency
        min_val = np.nanmin(self.augmented_data)
        max_val = np.nanmax(self.augmented_data)

        normalized_arr = 1 - (self.augmented_data - min_val) / (max_val - min_val)


        #sigmoid function NEED TO ADD OPTIONS WHETHER TO USE SIGMOID FUNCTION OR NOT 
        normalized_arr = 1/(1+np.exp((-(self.p*(normalized_arr-self.k*0.5)))))
        #converts nan values to 0 (makes transparent)
        normalized_arr[np.isnan(normalized_arr)] = 0

        colored_data[:,:,3] = (normalized_arr*255).astype(np.uint8)
        
        img = Image.fromarray(colored_data)

        if (self.image_number == 0 or self.image_number == 2):
            img.save('image/live1.png') 
            self.image_number = 1       
        else:
            img.save(f"image/live{self.image_number + 1}.png")
            self.image_number = 2

    def end_loop(self):
        self.loop_indicator = 0

    def loop_status(self):
        return self.loop_indicator

    def get_img_no(self):
        return self.image_number

    def update_p(self, p):
        if (p<127):
            new_p = p/10
        else: new_p = p

        self.p = new_p    

    def update_k(self, k):
        new_k = k / 255 * 2
        self.k = new_k

    def update_k_and_p(self, k, p):
        self.k = k
        self.p = p

    def output_image(self): 
        pass
        #self.output_image = Image.fromarray(self.image_array,mode = "RGBA")
        #self.output_image.show()

    def change_colourmap(self, map):
        self.colourmap = map

    def value_selector(self, selector):
        if (selector):           #real numbers
            self.real_selection = True
            self.augmented_data = self.raw_data_array[:,:,0]
        else:                                #imaginary numbers
            self.augmented_data = self.raw_data_array[:,:,1]
            self.real_selection = False

    def return_k (self):
        k = self.k * 255 / 2
        return k

    def return_p (self):
        if (self.p<127):
            p = self.p*10
        else: p = self.p
        return p
          
image_plotter = plotter()
Superobject = Superclass()

PASSWORD = 'Puzzlebot72'

def plot(image_plotter, Superobject, pathlength = 960, timesteps = 50):

   #forward iterations between turns
    f = 906
    #turn iterations 
    t = 79
    #one sequence = forward, left, forward, right
    no_seq = 19
    #start / end iterations
    se = 29
    #x forward
    xf = 0.833155

    #total_size 
    total_it = no_seq*2*(f+t) + 2*se - t

    #total_it = se+2*f+2*t + 1

    #geometry 
    #turn radius #may need to be different for left and right 
    r_l = 10.5
    r_r = 9

    #centre offset of coils
    adjacent_sep = 0.75


    #initialising path
    coil_X = np.zeros(total_it)
    coil_Y = np.zeros(total_it)
    coil_THETA = np.zeros(total_it)

    #initialise turn centers. left and right. 
    l_centre = np.empty((no_seq,2))
    r_centre = np.empty((no_seq,2))

    for j in range (no_seq):

        l_centre[j,:] = [779,10.5 + 42*j]
        r_centre[j,:] = [21 ,30 + 42*j]

    #print(l_centre)
    #print(r_centre)

    #first forwards pass

    #initial forward movement 
    for i in range (se):
        coil_X[i+1] = coil_X[i] + 0.833155
    coil_Y[0:se+1] = 0
    coil_THETA[0:se+1] = 0

    for j in range (no_seq): 
        # start indexes: 
        fsi = (se+j*2*(f+t))
        lsi = fsi + f
        bsi = lsi + t
        rsi = bsi + f
        
        #forward
        for i in range (fsi-1,lsi): 
            coil_X[i+1] = coil_X[i] + 0.833155
            coil_Y[i+1] = coil_Y[i]
            coil_THETA[i+1] = coil_THETA[i]

        #coil_Y[fsi:lsi+1] = coil_Y[fsi-1]
        print('forward',j)
        
        #left 
        for i in range (lsi-1,bsi): 
            coil_THETA[i+1] = coil_THETA[i] + np.pi/t
        coil_X[lsi:bsi+1] = l_centre[j][0] + r_l*np.sin(coil_THETA[lsi:bsi+1])
        coil_Y[lsi:bsi+1] = l_centre[j][1] - r_l*np.cos(coil_THETA[lsi:bsi+1])

        print(l_centre[j][0])
        print('left',j)

        #backward 
        for i in range (bsi-1,rsi): 
            coil_X[i+1] = coil_X[i] - 0.833155
            coil_Y[i+1] = coil_Y[i]
            coil_THETA[i+1] = coil_THETA[i]

        #coil_Y[bsi:rsi+1] = coil_Y[bsi-1]
        #coil_THETA[bsi:rsi+1] = coil_THETA[bsi-1]
        print('backward',j)

        #right
        if j == no_seq - 1: 
            break

        for i in range(rsi-1,rsi+t): 
            coil_THETA[i+1] = coil_THETA[i] - np.pi/t
        coil_X[rsi:rsi+t+1] = r_centre[j][0] - r_r*np.sin(np.pi - coil_THETA[rsi:rsi+t+1])
        coil_Y[rsi:rsi+t+1] = r_centre[j][1] - r_r*np.cos(np.pi - coil_THETA[rsi:rsi+t+1])
        print('right',j)
   
    #final movement
    print(coil_X[total_it - se - 1])
    print(coil_Y[total_it - se - 1])
    print(coil_THETA[total_it - se - 1])

    for i in range (total_it - se, total_it):
        coil_X[i] = coil_X[i-1] - 0.833155
        coil_Y[i] = coil_Y[i-1]
        coil_THETA[i] = coil_THETA[i-1]
    
    #geometric transform 
    #geometric transform 
    robot_THETA = coil_THETA[:]

    robot_X = coil_X[:] + np.sin(-coil_THETA)*adjacent_sep
    robot_Y = coil_Y[:] + np.cos(-coil_THETA)*adjacent_sep
  

    #initialisation process 
    Superobject.find_inital_origin()

    st = time.time()

    #update in real time version
    #counter = 0

    #timesteps = 20
    #for j in range(timesteps):
    #    t = j*pathlength/timesteps
    #    for i in range(int(pathlength/timesteps)):
    #        k = int(t+i)
    #        Superobject.process(X[k],Y[k],THETA[k])
    #        Superobject.sensor_data.increment_data()
#
    #    raw_data_array = Superobject.output_array()
    #    image_plotter.newinput(raw_data_array)
    #    image_plotter.raw_data_ops("real","jet")
    #    #time.sleep((pathlength/timesteps)*0.01)
    #    #output image in this one - call image function

        
    #original all rasterization version
    #for i in range(pathlength):
    #    Superobject.process(X[i],Y[i],THETA[i])
    #    Superobject.sensor_data.increment_data()
    for i in range(total_it - 1):
        Superobject.process(robot_X[i],robot_Y[i],robot_THETA[i])
        Superobject.sensor_data.increment_data()
        #print('starting',i)
    #Superobject.process(x,y,theta)
    #Superobject.process(x2,y2,theta2)

    et = time.time()
    te = et - st

    print("time elapsed: ",te," seconds")

    raw_data_array = Superobject.output_array()
    #Superobject.plot()

    #image stuff
    image_plotter = plotter()
    #
    image_plotter.newinput(raw_data_array)
    image_plotter.raw_data_ops("real","jet")

    image_plotter.output_image()

    while (image_plotter.loop_status() == 1):
        image_plotter.regenerate()

def copy_image(source_path, destination_path):
    try:
        shutil.copyfile(source_path, destination_path)
        return True
    except Exception as e:
        print(f"Error copying file: {e}")
        return False
    
def authenticate():
    return Response('Please enter the password.', 401,
                    {'WWW-Authenticate': 'Basic realm="Password Required"'})

def unauthorized():
    return Response('Unauthorized Access', 403)

def requires_password(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or auth.password != PASSWORD:
            return authenticate()
        return f(*args, **kwargs)
    return decorated

@app.route('/')
@requires_password
def index():
    return render_template('homepage.html')

@app.route('/update_k', methods=["POST"])
def update_k():
    k = int(request.form["k_slider"])
    image_plotter.update_k(k)
    return "k updated"

@app.route('/end_loop')
def end_loop():
    global image_plotter
    image_plotter.end_loop()
    return "loop ended"

@app.route('/update_p', methods=["POST"])
def update_p():
    p = int(request.form["p_slider"])
    image_plotter.update_p(p)
    print("p:" + p)
    return "p updated"

@app.route('/display_plot')
def display_plot():
    global image_plotter

    if image_plotter is not None:

        img_number = image_plotter.get_img_no()

        if img_number == 0 or img_number == 12:
            return send_file("image/transparent.png", mimetype='image/png')
        else:
            return send_file(f"image/live{img_number}.png", mimetype='image/png')
    else:
        return send_file("image/transparent.png", mimetype='image/png')  

@app.route('/display_test_piece')
def display_test_piece():
    return send_file("image/blank.jpg", mimetype='image.jpg')

@app.route('/display_transparent')
def display_transparent():
    return send_file("image/transparent.png", mimetype = 'image/png')

@app.route('/plot_function')
def plot_function():
    global image_plotter, Superobject
    
    plot(image_plotter, Superobject)
    return

@app.route('/real_or_img', methods=['POST'])
def real_or_img():
    global image_plotter  
    data = request.get_json()

    real_imaginary = data.get('real_imaginary') 

    if real_imaginary == 'real':
        image_plotter.value_selector(1)
    else:
        image_plotter.value_selector(0)

    
    return "Values Used Updated"

@app.route('/handle_thresholding', methods=['POST'])
def handle_thresholding():
    global image_plotter
    
    data = request.get_json()
    thresholding = data.get('thresholding') 

    if thresholding == 'defects_light':
        image_plotter.update_k_and_p(0.5, 150)
    elif thresholding == 'defects_heavy':
        image_plotter.update_k_and_p(1.25, 150)    
    elif thresholding == 'transparent_entire':
        image_plotter.update_k_and_p(0, 0)
    elif thresholding == 'transparent_non_defects':
        image_plotter.update_k_and_p(0.1, 170)
    elif thresholding == 'entire':
        image_plotter.update_k_and_p(0, 150)

    k = image_plotter.return_k()
    p = image_plotter.return_p()
    print(k)
    print(p)

    return jsonify({'k': k, 'p': p})

@app.route('/select_colourmap', methods=['POST'])
def select_colourmap():
    global image_plotter  
    data = request.get_json()

    colourmap = data.get('colourmap') 

    image_plotter.change_colourmap(colourmap)
    
    return "colourmap Updated"

@app.route('/save_image', methods=['POST'])
def save_image():

    global image_plotter
    img_number = image_plotter.get_img_no()
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    destination_filename = f"test_piece_{timestamp}.jpg"

    source_path = os.path.join(app.root_path, f"image/live{img_number}.png")
    destination_path = os.path.join(app.root_path, 'image', 'generated', destination_filename)

    copy_image(source_path, destination_path)
    return "Image Saved"
   

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
