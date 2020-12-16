### Word Prediction with LSTM-RNN using Keras + Tensorflow

---------------------------------
How to use :

Suppose we have two input files over which we wish to implement a next-word prediction algorithm. Arrange the files as given in the repository and run the commands as shown in [howto.txt](https://github.com/deeptavker/Word_Prediction_RNN/blob/master/howto.txt). Note that we can have any arbitary number of files. 

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
2. In addition to normal one word prediction, assume that the text files have **tagged** sentences which is often found in legal documents, for example, a paragraph which contains specific arguments related to some specific section of the *law* would be tagged according to the label for that *law* such as *1A*. While prediction, the user can input such specific section and the next word prediction algorithm should modify the output according to the new probability distribution. 
3. We also want the feature of continuous learning which enables the algorithm to continually learn new words as the user interacts with the application. 
4. Implement a reliable accuracy testing metric which can be used to evaluate and optimise the performance of the algorithm  and can also be used as a benchmark while marketing the application to consumers. 
5. Implement batch processing which ensures that irrespective of the combined size of input files, the computer memory is able to process the data without facing a memory overload. 

### Solution Approach

1. We use an LSTM based Recurrent Neural Network for tackling this problem. 
2. We pad the training data by the section for each input instance for modified learning. And the same tag can be used while prediction. 
3. Implement Elastic Weight Consolidation technique developed by DeepMind.AI
4. We use N-Gram accuracy as a metric for testing. 
5. We use data generators provided by the **Keras** library for ensuring memory problems don't occur. 

### Solution Algorithm

#### 1.1 Making the data trainable 

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

#### 1.2 The Neural Network

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

#### 1.3 N-Gram Accuracy

We generate N-Grams for accuracy testing. We run predictions on the first part splitted N-Grams data and compare it to the last word in that N-Gram. Using plain averaging, we calculate the correctness of predictions over a large number of such N-Grams. 

In this code, I used 4-grams. Here's the accuray metric calculated over both input files.

`4-Gram Accuracy = 0.991214470284`

#### 1.4 Tagged data prediction

When we have tags for various sentences in the input file. We use those tags to pad the input sequences generated by decomposing the input files by a particular number corresponding to that tag (just like tokenization). For example

- Tag 1 -> A B C D 
- Tag 2 -> F G H I 
- For a window size of 4, the I/O becomes : 
- 1 + tokenized(A B C) -> one-hot encoded (D) 
- 2 + tokenized(F G H) -> one-hot encoded (I)

We use these same tags during prediction. We set up the implementation such that no tag provided is the same as all tags provided. We can also supply multiple tags during prediction, the implementation will handle that. 

#### 1.5 Batch Processing

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

#### 1.6 Continuous Learning

The main objective is to avoid *catastrophic forgetting* in our neural network. We use **Elastic Weight Consolidation (EWC)** for continuous learning. [Here](https://arxiv.org/pdf/1612.00796.pdf) is the research paper for reference. We calculate the Fisher Matrix like so. 

![Fisher](/pics/fisher.jpeg)

Once we have the fisher matrix, after learning one task, we know what parameters are important for retaining performance for that particular task. Therefore, we introduce a quadratic penalty in the loss function which makes these particular parameters elastic, i.e. resistant to change in value. The modified loss function looks like so. 

![loss](/pics/loss_ewc.png)

We re-calculate star-variables and the fisher matrix after each task. 

I impelmented EWC using both keras and tensorflow. This feature is still in development phase and is not completely accurate. But the code provided in `adaptive_train.py` and `adaptive_2_train.py` is a good place to start for anyone who is looking to implement EWC.  

### Solution Implementation

#### 1.1 Normal word prediction with Batch Processing 

![Flowchart](/pics/flowchart.png)

#### 1.2 Tagged word prediction 

The user will have to modify compute_vocab for this feature. Also, input sequence length should be changed wherever necessary. 

#### 1.3 Adaptive word prediction

![adaptive](/pics/adaptive.png)



NOtes

- Two input files generated using lipsum.com each of 2000 words and 12KB each. 



