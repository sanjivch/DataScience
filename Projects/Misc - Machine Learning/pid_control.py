import numpy as np 
from scipy import signal
import streamlit as st 
import control
import matplotlib.pyplot as plt

st.title('PID Controller')

'''Process Parameters'''
# Transfer function - Process

K = float(st.text_input("Process gain","3"))
T = float(st.text_input("Time constant","4"))
num_p = np.array([K])
den_p = np.array([T , 1])
Hp = control.tf(num_p, den_p)

'''Controller Parameters'''
# Transfer Function - PI Controller
Kp = float(st.text_input("Proportional gain","0.4"))
Ti = float(st.text_input("Integral time","2"))
Td = float(st.text_input("Derivative time","0"))
num_c= np.array([Kp*Ti, Kp])
den_c= np.array([Ti, 0])
Hc= control.tf(num_c, den_c)
print(Hc)

# The Loop Transfer function
L = control.series(Hc, Hp)
print ('L(s) =', L)

# Tracking transfer function
T = control.feedback(L,1)
print ('T(s) =', T)

# Step Response Feedback System (Tracking System)
t, y = control.step_response(T)
    

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.plot(t, y)

ax.set_xlabel("t")
ax.set_ylabel("y")    


ax.set_title("Step Response Feedback System T(s)")
ax.grid()

st.write(fig)
