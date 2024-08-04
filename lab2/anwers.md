## 1. Download the simple self attention example from the lecture, here Download here, and code up an example of the sentence "boy bites dog". Code the example so that each embedding is a unit vector of 3 dimensions and that 'bites' shades  both 'boy' and 'dog'. (correction, you can drop the word 'unit' from previous sentence) 

Normalized Output Embeddings :
tensor([[    0.93,     0.05,     0.02],        
        [    0.00,     1.00,     0.00],        
        [    0.01,     0.07,     0.92]])  

## 2. Add code to normalize the output embeddings to unit vectors. Verify again that the bites is shading final embeddings of both 'boy' and 'dog', however slightly.
```Python
New/torch.sum(New, axis=1, keepdim=True)
```


## 3. Discuss how your example handles "dog bites boy". Is this adequate? If not, how should we address this? 
It handles it the same as "boy bites dog" as it is a bag of words model. To fix this we need to add some sort of positional information.


 ## 4. Download the Transformer Code here Download here, study the code and then train for at least 10 epochs. Sorry this may take awhile (correction, just run for 1 epoch if you must.  Getting lots of reports that many students' hardware is not up to task) (About 30 times faster on a GPU).

Done

## 5. Compare inferences from your 10 epoch model with Michael's 79 epoch model here 
* SOURCE: Harris said it was old.
* TARGET: Harris sade, att det skämtet var gammalt.
* PREDICTED 10: Harris sade det var det .
* PREDICTED 80: Harris sade , att det skämtet var gammalt .
---
* SOURCE: Harris and I appeared to be struck by it at the same instant.
* TARGET: Harris och jag verkade slås av den i ett och samma ögonblick.
* PREDICTED 10: Harris och Harris och jag tog föreföll  var på ett tag på den . 
* PREDICTED 80: Harris och jag verkade slås av samma ögonblick till ett ögonblick .
---
* SOURCE: It must have been worth while having a mere ordinary plague now and then in London to get rid of both the lawyers and the Parliament.
* TARGET: Londonborna måste ha ansett det vara ett billigt pris, detta att då och då drabbas av pesten, för att bli av med både advokater och parlamentsledamöter.      
* PREDICTED 10: Det måste ha varit en som varit en man som en  och då han  och  och  och .
* PREDICTED 80: måste ha ansett det vara ett billigt pris ’ vara ett billigt då detta då som då som för av.
---
* SOURCE: "Don't go to sleep, old man," we said as we started.
* TARGET: ”Somna inte bara, gamle gosse”, sade vi, då vi gick iväg.
* PREDICTED 10: ”  inte sade vi kunde vi kunde , ” vi ”, sade , då vi . 
* PREDICTED 80: ”  inte bara vi kommer talar gamle gosse ”, sade , då vi gick vi gick vi gick .
--- 
* SOURCE: What time is it? 
* TARGET: Vad är klockan?
* PREDICTED 10: Vad Vad är det är ?   
* PREDICTED 80: Vad är det ?  

## 6. Download and run either minGPT Links to an external site. or nanoGPT Links to an external site. over the Shakespere example.
```
KING RICHARD III:
The law bloods and boast we are ruled beats,
Unless that I will be traitor.

HareNARD:
Provost, take him the other to thy heart my heart.

KING RICHARD III:
Stir the order, and the king!

SWARD III:
How now, these wondrous state are the tyrants of the foot?
And so doth thou mother'st Warwick thy highness.

Father:
Then, in France again; the thrall thrust for the great shadow
A flesh to fail or death? What hour hast thou a
Radam? then I do bid him to mark no more sole
Than thou
---------------

Men passage me, my soul was my master and several;
The world makes me starve the table ground.

JULIET:
I'll stay the lord
Is as for the journey of Paulina's straight.

ROMEO:
And I cannot stay the hour of Rome,
In this Claudio should be in Rome. For my heart's babe
That shall now a thousand of love.

ROMEO:
Then, this is come I say; for I know thee;
In the external of death is determined.

JULIET:
Sprrtake it, and give her blood me sleep, on some meaning,
And all toil, would I will; you were no
---------------

Men that I boys. Then I'll say it will not
save the prize tale of one actions, and
to be rest mine: so men so, I would stand the king!

HERMIONE:
The beggar-harm of his hands in the particular and keep
He was a self-sentenced bird again. What evil I,
Have deserved for Himself and Bluntlophet's colourse,
To bag him of the former and bear some others state,
All tradens of the value the bloody war,
And with the gates of babes his moaner to that our children years,
And bring him out of all hollow th
---------------

The seas of my soul is the sea, and I fear
The accompany cords of my behalf and full of
My brother, who drinks of my Lord.

HENRY BOLINGBROKE:
My master, unlawful farewell, that becomes him,
And as I do suppose I have blazen yet:
But I fly, go with you to require! the harket is ours.

DUKE OF YORK:
What are you common with the people's life?

DUCHESS OF YORK:
If you will win all no sword, proud he is a word.

DUCHESS OF YORK:
No, nor I: pardon to swear a whore.

DUCHESS OF YORK:
You have a matte
---------------

BUCKINGHAM:
Then, here comes come to the ground I have been
As to-morrow more have brought in no doubt,
Who shall I take him I love to the deed for as
I do they swim together out of him to as you.
That I say, you will not stay and ne'er speak.

KING RICHARD II:
Madam, that I until some soar is it now;
Or her which I would toucher his soul.

CATESBY:
If e'er your mother, the sea of double grace way.

DUKE OF YORK:
Unless move but his highness, and that God,
More but we do more than mean with self
---------------


MENENIUS:
He hath had not a sing he that take him in his
A peace of his person of him.

MARCIUS:
How to you? what think'st thou?

CORIOLANUS:
No.

Messenger:
Never the consul, stay with her good close? What of that?
Come her? what do you not?

MARCIUS:
Here's a countrymation; an here better than what he would
Becomes you think of it.

CORIOLANUS:
Call your cheek; and, and not though what he is so,
With Corioli in Paulina,
His cruel in his tempts by the grave blows of blows,
Even in the changes
---------------

She would be made from thee, fie! Not that he sinks she:
The way is not say, though the way are although
I have a party storms. But in the oath, a sick,
Not time to have of thee, as that woman's true is,
To fight the voice, whom we have done to the warlike,
And unrancing the arse night the world!
Then they say the chequer of the time.

DUKE VINCENTIO:
Torment them of the speech of his taunt.

DUKE VINCENTIO:
O, that thou wert as a art thou as wouldst carry to xolicio,
Here in our hopes that will
---------------

I'll play them again. And, ere't thou art was
A damned sick to the king, to order that thou lovest.

First Citizen:
A never growth other best to him. This was known
To use your guest to pieces,--bow your sword
Must be over-babes, whereon you lies his love,
And his sign breasted strange, there do open
A pair of point and your copes have person'd.

CORIOLANUS:
Pray you, poor soul I think,
And though you must be his counsellor beseech your highness.

BRUTUS:
Love her, nobleman, and love him call th
---------------


LEONTES:
The realm be strange on another, the rools,
I know not not the valiant of the tail;
Poor soul, but my soul!

PAULINA:
You shall be so.

LEONTES:
Go you before you.

PAULINA:
But it is not the manner of my soul,
Is not mannerror to obey yourselves. O, with
And be your oily noble face and death
To contempt as you'll depute your forger.

CAMILLO:
Madam, sir, I hold you to your happings and your words.

CAMILLO:
Even her brother, the hate at suppose your highness had stern'd
with the beast
---------------

How cheer to rather and redress him our gracious lord?

KING HENRY VI:
I say he two in the king, Buckingham, and you
His brother so patricians that makes him so fast,
Not so, between our back: pray you, I do will ridge,
You send you on your love.

GLOUCESTER:

CLARENCE:
And love-morrow your ladyship.
Marry, look you, Lord Angelo, no more more.

FLORIZEL:
He hath done of me.

LADY CAPULET:
Not till I say he speak you, but mounted to his grace.

Nurse:
So far--

Nurse:
The matter how I did speak n
---------------
```

## 7. In your own words, why do we need to stack the decoder blocks so deep?

It is important to have a deep decoder block to be able to get all of the complex relationships that we have in a sentence. The extra complexity also makes it able to understand long distance relations.

## 8. In your own words, why do we need residual connections?
Residual connections are that the input vector is inputed to multiple layers and not just the first. This is needed so we do not get vanishing gradients in the back-prop step. This means that we can have deeper networks with more blocks. If we have a residual connection where the input always is added the gradient will always be of magnitude 1. It also "reminds" the transformer network what it should be based on.

## 9. In your own words, why do we call GPT 'auto-regressive'?

It is called autoregressive because the output becomes the input so it generates text based on the previous text it has generated.

## 10. Install a python callable Llamma2 on your own computer. Note you will use this in subsequent recitations

```bash
ollama serve
ollama run llama2
```