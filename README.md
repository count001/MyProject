## My project

A Go AI mainly implements simple neural network and Monte Carlo tree search algorithms.

In addition, the file also includes tables and images for the evaluation chapter.

### How to use

Install Python related packages

```bash
pip install torch numpy sgfmill
```

Obtain and process data: https://homepages.cwi.nl/~aeb/go/games/games.7z. 

unzip the file. 

You can directly use the provided model weights: policyNet.pt, playoutNet.pt,valueNet.pt.

Simply clone the above files into the same file and follow the steps below.

Run with GTP protocol

> You can use  [Sabaki](https://github.com/SabakiHQ/Sabaki) to deploy:
>
> when you finish install sabaki, open it and select ```Engines -> Manage Engines```
>
> Set the path as shown in the following figure:
>
> Click the ```add``` button in the bottom left corner to add the model.
> 
> Write the name of the model on the first line.
> 
> !!!Do not fill in the path on the second line, write ```Python``` instead.
> 
> Write the path of the ```gtp.py``` file in the folder on the ```third``` line!
>
> After completing these, click ```close```.
>
> ![image](https://github.com/count001/MyProject/blob/master/image/2024-08-05%20082940.png)
>
> Click on the menu in the bottom right corner and select ```'info'```.
> After that, select the opponent engine on the page that appears, you can set it as balck or white (in figure is white), and you can start the battle. As shown in figure:
>
> ![image](https://github.com/count001/MyProject/blob/master/image/2024-08-05%20083907.png)
>
> After that you can enjoy the game:
>
> ![image](https://github.com/count001/MyProject/blob/master/image/2024-08-05%20084007.png)


You can also run it directly from the command line (without interface):
```bash
python gtp.py
```

### References

+ [Mastering the game of Go with deep neural networks and tree search](https://www.nature.com/articles/nature16961)

+ [Sabaki](https://github.com/SabakiHQ/Sabaki)

+ [sgfmill](https://github.com/mattheww/sgfmill)

  
