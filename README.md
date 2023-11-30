# Voice-authentication-using-audio-modality

<b>Intro</b>
The goal of the study was to create an artificial intelligence model that allows users to log in using their voice. The used model is based on convolutional networks and a triplet loss function. The model, during training, learns to distinguish the voice of different individuals. The used database was LibriSpeech, which is a publicly available database containing hundreds of hours of high-quality sound recordings. The recordings include more than two thousand people, whose samples have been divided into fragments of four seconds each. In the learning process, the data were converted into mel spectrograms. For the test data, the model achieved a sensitivity of 65% and a 95% specificity. A high specificity value is important for this type of application. The model should maximize specificity, even at the expense of sensitivity, to prevent an unwanted person from logging in. In addition to the trained model, a mobile application was developed to implement the entire system. It allows the user to register and log in using voice.

<b>Model</b>
The network was created using the Keras module. The model consists of three identical parallel parts. Each contains nine convolutional layers, four max pooling layers, a Flatten layer and 3 fully conected layers. The total architecture of one part is shown in Table I. The activation function of the convolutional layers is leaky ReLU, and the activation function of the dense layers is ReLU. The max pooling mask had a size of two by two. The step for the convolutional layers was equal to one. Each part with this architecture allowed to extract a vector of one hundred features from the input. Then three vectors with one hundred features each were analyzed through a loss function.
<br>
![image](https://github.com/szymi999/Voice-authentication-using-audio-modality-with-deep-learning/assets/52047025/4faac4da-3af8-4170-9339-91119a509da4)
<br>

<b>Results</b>
Satisfactory results were achieved for triples formed as described above. Figure shows the confusion matrix. The overall accuracy of the model was 92%, specificity was 95%, and sensitivity was 65%. In this case, low precision is not a problem. It can be interpreted that on two out of three attempts, the user manages to log in correctly. The high specificity shows that the probability of bad authentication is low. The EER was used to select the threshold that determined whether the samples were from the same person. This is the value for which the graphs of the ratios of erroneously accepted and erroneously rejected samples intersect. Figure shows the determination of the EER. This yielded an EER value of 4.185% for a threshold of 99.1%. For testing, this result was rounded to 99%. The results shown in Figure 3 were achieved for such a threshold.
<br>
![image](https://github.com/szymi999/Voice-authentication-using-audio-modality-with-deep-learning/assets/52047025/573f9d0c-a5cf-43c0-a007-6db7ce66718f)
<br>
![image](https://github.com/szymi999/Voice-authentication-using-audio-modality-with-deep-learning/assets/52047025/5d72a0d5-80d1-4cf6-823d-d45ffc97eac4)
<br>

<b>Application</b>
One of the elements of the implemented project was to create a mobile application. For this task the Android Studio environment was used. The main tasks of the application were to enable registration of new users and logging in by voice. For this purpose, the necessary graphical user interface was created. During the registration process, the application requires the user to enter a user name and record three voice samples of two seconds each. The sound recordings are stored in the device's memory, and the access paths to the recordings are stored in a SQLite database along with the user's name. The database is also stored in the device's memory. The login process requires the user to enter their name and record one voice sample. Based on the name given, the application extracts the recordings stored during the registration and, using a trained model, compares the stored recordings with the one given during the login. If the recordings turn out to be similar enough, the user is informed that the operation was successful, otherwise he receives information that the login was not successful.
<br>
![image](https://github.com/szymi999/Voice-authentication-using-audio-modality-with-deep-learning/assets/52047025/333d2f6a-3f88-46bc-a04c-43f0cccd305f)
