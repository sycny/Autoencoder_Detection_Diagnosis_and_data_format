There are two folders:One is for the Autoencoder detection and diagnosis. Another one is for the data format.

Folder One:
It contains three parts:
1. The first part is to convert the raw data to PMU data(with limit and without limit)
2. The second part is about autoencode based anomaly detection.
3. The third part is about autoencode and kmeans based anomaly diagnosis.
The part two and three use the PMUalldatanpz.npz file.

Folder two:
The Raw wave only contains the PCC node information. 7 converters data are not included due to its large size.
The PMU data contains the 7 converters data.

All the data are organized in the npz.file.

The array 'a' in the npz.files is time series data.
The array 'b' in the npz.files is the label data.
There are 56 cases in the train set. 8 cases in the test set. 62 cases in the all set(not included the two normal cases)