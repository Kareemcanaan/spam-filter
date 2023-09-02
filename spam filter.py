import pandas as pd
import numpy as np

# Sample data containing emails and labels (1 for spam, 0 for not spam)
data = {
    'email': [
        'Buy cheap watches!', 
        'Work part time without visa',
        'Congratulations, you won $1,000,0000',
        'Are you getting my email messages?',
        'Hope you got the email that I sent to you yesterday'
    ],
    'label': [1, 0, 1, 0, 1]
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# List of spam keywords
spam_keywords = ['buy', 'cheap', 'won', 'free', 'offer']

# Function to check if an email contains spam keywords
def is_spam(email):
    for keyword in spam_keywords:
        if keyword in email.lower():
            return 1
    return 0

# Apply the spam filter function to each email
df['predicted_label'] = df['email'].apply(is_spam)

# Compare predicted labels with actual labels
df['correct'] = np.where(df['predicted_label'] == df['label'], 'Correct', 'Incorrect')

# Print the DataFrame
print(df)