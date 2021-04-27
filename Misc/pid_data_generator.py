import streamlit as st 
import numpy as np 
from scipy import signal 
import streamlit as st
import control
import matplotlib.pyplot as plt
import pandas as pd
import random
import datetime


def get_transfer_function(process_params, controller_params, dead_time = False):
    '''
    get_transfer_function [summary]

    [extended_summary]

    Args:
        process_params ([type]): [description]
        controller_params ([type]): [description]
        dead_time (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    '''

    # Transfer function - Process
    num_p = np.array(process_params[0])
    den_p = np.array([process_params[1], 1])
    Hp = control.tf(num_p, den_p)

    # Transfer Function - PI Controller
    num_c= np.array([controller_params[0]*controller_params[1], controller_params[0]])
    den_c= np.array([controller_params[1], 0])
    Hc= control.tf(num_c, den_c)

    # Transfer Function - Series
    L = control.series(Hc, Hp)

    # Transfer function - Closed loop
    T = control.feedback(L,1)

    if dead_time:
        # Generating transfer function of Pade approx :
        theta = float(st.sidebar.text_input("Dead time","2"))
        
        n_pade =10
        (num_pade, den_pade) = control.pade(theta, n_pade)
        H_pade = control.tf(num_pade, den_pade)

        # Generating transfer function with time delay :
        T = control.series(H_pade, T)
        print(T)
    return T
    
def set_input_data(t_initial = 0, t_final =  86400, dt=1, _amplitude = 5):
    '''
    set_input_data [summary]

    [extended_summary]

    Args:
        t_initial (int, optional): [description]. Defaults to 0.
        t_final (int, optional): [description]. Defaults to 86400.
        dt (int, optional): [description]. Defaults to 1.
        _amplitude (int, optional): [description]. Defaults to 5.

    Returns:
        [type]: [description]
    '''
    #  Defining signals 
    t_initial = 0
    t_final = 86400
    dt = 1 #sec
    num_points = int(t_final/dt) + 1 # Number of points of sim time
    
    u = list()
    
    #Intial u
    _amplitude = 5 

    # Intialize counter - this will be used to hold the stepped value until its next change
    counter = 0
    time_horizon = np.linspace(t_initial, t_final ,num_points)
    for i in time_horizon:
        
        # Everytime counter resets to 0, a "random" input is generated 
        # The "random" input will be held until the counter changes
        # A randomized movement in input is achieved this way
        if counter == 0:
            # Change the input within 75 to 150 minute window
            counter = random.randint(4500, 9000)              
            
            # Generate random input
            value = random.randint(0, 100*_amplitude)/100        
            
        # Append the same random input (u) until the counter resets
        if counter > 0:
            u.append(value)
                            
        # Decrement the counter to change the input
        counter = counter - 1

    return u, time_horizon, num_points

def generate_response(T, t, u, X0 = None):
    '''
    generate_response [summary]

    [extended_summary]

    Args:
        T ([type]): [description]
        t ([type]): [description]
        u ([type]): [description]
        X0 ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    '''
    t, y = control.forced_response (T, t, u, X0 =X0)
    return t, y

def add_noise(y, num_points):
    '''
    add_noise [summary]

    [extended_summary]

    Args:
        y ([type]): [description]
        num_points ([type]): [description]

    Returns:
        [type]: [description]
    '''
    # Add noise to make the response realistic
    return y + np.random.normal(0,0.05,num_points)

def convert_to_dataframe(_time, _input, _output):
    '''
    convert_to_dataframe [summary]

    [extended_summary]

    Args:
        _time ([type]): [description]
        _input ([type]): [description]
        _output ([type]): [description]

    Returns:
        [type]: [description]
    '''
    
    data = pd.DataFrame()
    data['Timestamp'] = _time
    data['Input (SP)'] = _input
    data['Response (PV)'] = _output

    a = datetime.datetime(2021,4,22,00,00,59)
    b = a + datetime.timedelta(minutes=1)
    print(a,b)
    data['Timestamp'] = pd.date_range(pd.to_datetime('today', format='%Y%m%d %H:%M:%S'), periods=86401, freq='1min')
    return data

def export_as_csv(data, file_name=None):
    '''
    export_as_csv [summary]

    [extended_summary]

    Args:
        data ([type]): [description]
        file_name ([type], optional): [description]. Defaults to None.
    '''
    data.to_csv('PID_train_data.csv', index=False)

def plot_data(t, u, y):
    '''
    plot_data [summary]

    [extended_summary]

    Args:
        t ([type]): [description]
        u ([type]): [description]
        y ([type]): [description]
    '''
    fig = plt.figure (1, figsize =( 15, 8))
    plt.plot (t, y, 'blue')
    plt.plot (t, u, 'green')
    plt.xlabel('Time')
    plt.ylabel('PV, SP')
    plt.grid()
    plt.legend(labels =( 'Response (PV)', 'Input (SP)'))
    st.write(fig)

# ===============================================
st.title('PID Controller Data Generator')

st.button("Export as CSV")

st.sidebar.title('Parameters')

st.sidebar.subheader('Controller Parameters')
Kp = float(st.sidebar.text_input("Proportional gain (Kp)","0.4"))
Ti = float(st.sidebar.text_input("Integral time (Ti)","2"))
Td = float(st.sidebar.text_input("Derivative time (Td)","0"))

if_dead_time = (st.sidebar.checkbox('Apply dead time in response'))


if (st.sidebar.checkbox('Show Process Parameters')):
    st.sidebar.subheader('Process Parameters')

    K = float(st.sidebar.text_input("Process gain (K)","3"))
    T = float(st.sidebar.text_input("Time constant (T)","4"))

else:
    K, T = 3, 4
    

process_params = [K,T]
controller_params = [Kp, Ti, Td]


u, time_horizon,  num_points = set_input_data()
T = get_transfer_function(process_params, controller_params, dead_time = if_dead_time)
t, y = generate_response(T, time_horizon, u, X0 =5)
y = add_noise(y, num_points)

data = convert_to_dataframe(t, u ,y)
plot_data(t, u, y)




st.subheader('Input vs Response')

st.dataframe(data)

st.success('Data generated successfully')

