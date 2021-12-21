# Dog-Breed-Classifier

This is the repository for the Dog breed classifier project from Udacity's DS Nanodegree program.

- Based on a photograph of a dog, the algorithm will determine its breed.
- If a photograph of a person is provided, the code will indicate the dog breed that most closely resembles the person.

Convolutional Neural Networks (CNNs) can be used to solve the multiclass classification problem.

The solution consists of three steps.
- To begin, we can use existing algorithms to recognise human photos, such as OpenCV's implementation of Haar feature-based cascade classifiers.
- Second, we'll use a VGG19 model that has already been trained to recognise dog images.
- Finally, after determining whether the image is dog or human, we can send it to a CNN model, which will analyse it and select a breed from a list of 133 breeds that best matches the image.


## The Road Ahead
We break the notebook into separate steps. Feel free to use the links below to navigate the notebook.

- Step 0: Import Datasets
- Step 1: Detect Humans
- Step 2: Detect Dogs
- Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
- Step 4: Use a CNN to Classify Dog Breeds (using Transfer Learning)
- Step 5: Create a CNN to Classify Dog Breeds (using Transfer Learning)
- Step 6: Write your Algorithm
- Step 7: Test Your Algorithm
- 

## CNN's model was created from the ground up.

- I built a CNN model from the ground up to solve the problem. In the model, there are three convolutional layers.
- All convolutional layers utilise kernel size 3 and stride 1.
- The first conv layer (conv1) uses the 224*224 input picture, while the last conv layer (conv3) generates a 128 output size.
- The ReLU activation function is employed in this case.
- The pooling layer (2,2) is employed, which reduces the input size by 2x.
- two completely connected layers that provide a 133-dimensional output
- A 0.2 dropout is incorporated to prevent overfitting.

## Refinement - A CNN model was created using transfer learning.

- A CNN built from scratch has a 31 percent accuracy, but transfer learning can considerably improve the model.
- I picked the Resnet101 architecture, which is 101 layers deep and pre-trained on the ImageNet dataset, to build a CNN utilising transfer learning.
- The latest convolutional output of Resnet101 is used as an input in our model.
- We only need to add a completely connected layer to create a 133-dimensional output (one for each dog category).
- When compared to CNN built from the ground up, the model performed admirably. After only 20 epochs, the model achieved very good percent accuracy.


