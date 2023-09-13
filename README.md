# Ambulance-EC
Repository for the paper <i>"Emergency care for non-COVID-19 cases during and after the pandemic - A counterfactual study of response time of ambulance and transfer time to admit patients in hospitals"</i>

## Installing Requirements
The code is implemented in <i>Python 3.10.11</i>. Thus the system must have Python 3.10.11 installed.
If Python is installed, Clone the repository, and install the requirements using the following command in the terminal in repository directory:

```pip install -r requirements.txt```


## Directory Structure
The repo contains a ```utils``` directory which contains the following files:
- ```utils.py``` : Contains the code to both preprocess the data to be fed into the model and the post result processing.
- ```trainEach.py``` : Contains the code to train the model for each specific dataframe created for each emergency type
- ```trainAll.py``` : Contains the code to train the model for all the emergency dataframe created.
