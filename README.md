# Fake_news_Sentiment_model

I build a fakenews detection model utilizing NN layers with sentiment features and attention. Improving the tranditional fake news model, our model is adapted from DEAN model and HAN model.

To evaluate the effectiveness of each feature, we use quantitative analysis on each sentiment features. The result is shown below.

![image](https://github.com/alissa404/Fake_news_Sentiment_model/blob/main/ablation.png)


“ALL” denotes that all components in analyzer, including token frequency, arousal intensity, and sentiment patterns.
After removing each one of them, we obtain the sub-models “No-as”, “No-f”, “No-a” and “No-p”, respectively. “No-as” denotes that the experimental model has no sentiment features. “No-f” denotes that experimental model without token frequency, “No-a” denotes that without sentiment intensity, and “No-p” denotes that experimental model without sentiment patterns. 
