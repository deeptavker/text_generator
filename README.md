### Word Prediction with LSTM-RNN using Keras
---------------------------------
How to use :

Suppose we have two input files over which we wish to implement a next-word prediction algorithm. Arrange the files as given in the repository and run the commands as shown in [howto.txt](https://github.com/deeptavker/ME781-Course-Project/blob/master/howto.txt). Note that we can have any arbitary number of files. 

```sh
 Enter input text
-----------------------------------------------------------


Enter input string : lorem ipsum
+-----------+-------------+
| Next word | Probability |
+-----------+-------------+
|   dolor   |  0.64693826 |
+-----------+-------------+
-----------------------------------------------------------

Enter input string : --exit
(edm)bash-3.2$ 
```
_______

### Problem statement

1. Given a set of text files, develop and implement an ML algorithm which works as a predictive keyboard. Given a series of words, predict the next word based on the input text files. 
2. Implement a reliable accuracy testing metric which can be used to evaluate and optimise the performance of the algorithm  and can also be used as a benchmark while marketing the application to consumers. 
3. Implement batch processing which ensures that irrespective of the combined size of input files, the computer memory is able to process the data without facing a memory overload. 

### Solution Approach

1. We use an LSTM based Recurrent Neural Network for tackling this problem. 
2. We pad the training data by the section for each input instance for modified learning. And the same tag can be used while prediction. 
3. Implement Elastic Weight Consolidation technique developed by DeepMind.AI
4. We use N-Gram accuracy as a metric for testing. 
5. We use data generators provided by the **Keras** library for ensuring memory problems don't occur. 

### Solution Algorithm

#### Making the data trainable 

The basic idea is to virtually collate all the input data and decompose it into word strings of a fixed length using a sliding window technique. Suppose we use a windows size of 4. Let us represent the words as capital letters - A,B,C...

Window size : 4 => Input -> 3 & Output -> 1 (Always 1)

The following sentences gets decomposed like so. 

Original : A B C D E F G H

Decomposed :
- A B C -> D
- B C D -> E
- C D E -> F
... you get the idea. 

If windows size becomes 5, => Input -> 4 & Output -> 1

Original : A B C D E F G H

Decomposed :
- A B C D -> E
- B C D E -> F
- C D E F -> G
... and so on. 

Now, dealing with numbers in neural networks is more tractable than dealing with strings. So, what we do is we use tokenization in order to convert words into numbers. Tokenization just means we list all unique words in some order and assign whole numbers to each word. 

For the output we use one-hot encoding to represent the words. Let's see what that means. 
Consider the following tokenization
(A B C D E F G H) -> (0 1 2 3 4 5 6 7)
The output for sequence A B C -> D ; being the word *D* will be represented as the vector <0 0 0 1 0 0 0 0> 
This makes it easier for the neural network to predict the stuff we want to predict. It is also essential when we use a *softmax* layer in the neural network for the output. 

#### The Neural Network

Having the data ready to be trained, we supply it to a neural network with a number of LSTM, Dense and Softmax layers. It takes quite some time to train the data for text files of size > 1MB. 

```sh
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 3)                 0         
_________________________________________________________________
embedding_1 (Embedding)      (None, 3, 3)              561       
_________________________________________________________________
lstm_1 (LSTM)                (None, 3, 200)            163200    
_________________________________________________________________
lstm_2 (LSTM)                (None, 3, 200)            320800    
_________________________________________________________________
dense_1 (Dense)              (None, 3, 200)            40200     
_________________________________________________________________
dense_2 (Dense)              (None, 3, 187)            37587     
=================================================================
Total params: 562,348
Trainable params: 562,348
Non-trainable params: 0
_________________________________________________________________
```

#### N-Gram Accuracy

We generate N-Grams for accuracy testing. We run predictions on the first part splitted N-Grams data and compare it to the last word in that N-Gram. Using plain averaging, we calculate the correctness of predictions over a large number of such N-Grams. 

In this code, I used 4-grams. Here's the accuray metric calculated over both input files.

`4-Gram Accuracy = 0.991214470284`


#### Batch Processing

We use what are called data generator functions in order to make sure there are no memory overflows. The data generator functions makes sure to collect data from the hard memory, process it and then delete it from RAM and move on to the next chunk of data. This limits the maximum memory usage independent of the combined size of input files. 

```python
class Data_Generator(Sequence) :
    def __init__(self, fname, m) :
        self.indices = np.arange(0,m)
        self.fname = fname
        # actual name will be fname + _ + i/o + _ + idx + .npy ( idx -> 0 to m-1 )
        self.m = m
        # m is the number of files generated by pre-batching of npy files

    def __len__(self) :
        return m

    def __getitem__(self, idx) :
        temp_id = self.indices[idx]
        batch_x = np.load(self.fname + '_i_' + str(temp_id) + '.npy')
        batch_y = np.load(self.fname + '_o_' + str(temp_id) + '.npy')
        return (batch_x, batch_y)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.indices)
 ```


### Solution Implementation 

![Flowchart](/pics/flowchart.png)


### Model Training and Report

#### Python code styling and codebase information

- We have conformed to the PEP8 style guide conventions for our entire code base. Our code is also modularized into various utility based python scripts which can be individually modified for future customizations. 

- For memory stress testing, we used a 25MB text data file for training and used a memory profiling tool to see how much RAM is consumed while running the python program for each normal processing and batch processing. The batch processing algorithm automatically breaks it down into a specified size for the input-output process. We got good results for comparison. Bulk processing used around a maximum of 3GB of RAM whereas batch processing used around 350MB of RAM.  Also, for normal processing the RAM consumption increases with File Size whereas for batch processing it depends on the batch-size. For this case we used a batch size of 512. It was observed that batch processing is almost 10 times more memory efficient. Following two figures are for bulk and batch training memory consumption reports respectively. 

- Pics for memory analysis

- Training Log plots and details




### Notes

- Two input files generated using lipsum.com each of 2000 words and 12KB each. 



