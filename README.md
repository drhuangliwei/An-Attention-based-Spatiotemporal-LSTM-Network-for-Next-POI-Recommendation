# An Attention-based Spatiotemporal LSTM Network for Next POI Recommendation

Next POI (Point-of-Interest) recommendation, also known as a natural extension of general POI recommendation, is recently proposed to predict user’s next destination and has attracted considerable research interest. It focuses on learning users’ sequential patterns of check-in behavior and training personalized recommendation models using different types of contextual information. Unfortunately, most of the previous studies failed to incorporate the spatiotemporal contextual information, which plays a critical role in analyzing user check-in behavior, into next POI recommendation. In recent years, embedding learning and RNN (Recurrent Neural Network) based approaches show promising performance for modeling sequential patterns of check-in behavior in next POI recommendation. However, not all of the historical check-in records contribute equally to the next-step check-in behavior. To provide better next POI recommendation performance, we first propose a spatiotemporal long and short-term memory (ST-LSTM) network. By feeding the spatiotemporal contextual information into the LSTM network in each step, ST-LSTM can model the spatial and temporal information better. Also, we develop an attention-based spatiotemporal LSTM (ATST-LSTM) network for next POI recommendation. By using the attention mechanism, ATST-LSTM can focus on the related historical check-ins in a check-in sequence selectively using the spatiotemporal contextual information. Please refer to our paper An Attention-based Spatiotemporal LSTM Network for Next POI Recommendation for further details.

Next, we introduce how to run our model for provided example data or your own data.

# Environment

Python 3.5

TensorFlow 1.4.1

Numpy 1.15.0

# Usage
As an illustration, we provide the data and running command for Gowalla and Brightkite.

# Input data
userlocation.csv：includes user ID, POI ID, latitude, longitude, checkin time.

locations.csv: includes POI ID, latitude, longitude, city ID

The dataset can be downloaded from http://snap.stanford.edu/data/

# Contact
Liwei Huang, dr_huanglw@163.com

# Citation
If you use AT-LSTM in your research, please cite our paper:

Liwei Huang, Yutao Ma, Shibo Wang,Yanbo Liu. An Attention-based Spatiotemporal LSTM Network for Next POI Recommendation. IEEE Transactions on Services Computing. Accepted
