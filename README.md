# Analysis of Student Attrition in STEM Majors Using Representational Learning and Random Forest Modeling

Analysis of academic trajectory is a key component in assessing and improving retention in STEM majors.  A common practice when analyzing student academic behavior has been to generate analysis based off of human selected features.  While this approach can be effective when relating known phenomena, features that are crafted from domain knowledge potentially exclude discovery of useful, yet unfamiliar phenomena. In this work, a combination of modeling will be done to predict student retention in STEM.  The goal for the representational learning approach is that it will allow for the model to understand the architecture of course interconnectivity; and will be useful in other contexts.  The random forest will make predictions based off of the information provided from the representational learning, and additional student information.  The motivation for this combined modeling is to require as few assumptions about student behavior as possible, and instead allow prominent features of learner behavior to emerge organically.  

•	Skip-gram model meaningfully encodes courses into a learned vector representation.  Similar to NLP, the model will treat courses like words in a large corpus, and will learn the context of the courses within the corpus.   

•	The vector representation of course history will be the input to a random forest model (and possibly other models: logistic regression, neural network).
  o	Random forest will also take in additional information pertaining to gender, SES, academic preparation, course grades (and as a result, GPA), and grade anomaly.

•	Training is done with the skip-gram because the embedding cannot be learned live from the random forest model.
