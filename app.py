import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import plotly.graph_objects as go

def create_model():
    """Creates a simple model for CSAT prediction"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(42,)),
        Dense(32, activation='relu'),
        Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
    return model

def get_expected_columns():
    """Returns list of expected columns after one-hot encoding"""
    return [
        # Channel (3)
        'channel_name_Email', 'channel_name_Inbound', 'channel_name_Outcall',
        
        # Category (12)
        'category_App/website', 'category_Cancellation', 'category_Feedback',
        'category_Offers & Cashback', 'category_Onboarding related', 'category_Order Related',
        'category_Others', 'category_Payments related', 'category_Product Queries',
        'category_Refund Related', 'category_Returns', 'category_Shopzilla Related',
        
        # Sub-category (10)
        'Sub-category_Exchange / Replacement', 'Sub-category_Fraudulent User',
        'Sub-category_General Enquiry', 'Sub-category_Installation/demo',
        'Sub-category_Life Insurance', 'Sub-category_Missing',
        'Sub-category_Not Needed', 'Sub-category_Product Specific Information',
        'Sub-category_Reverse Pickup Enquiry', 'Sub-category_other',
        
        # Agent name (6)
        'Agent_name_Brenda Gillespie', 'Agent_name_Duane Norman',
        'Agent_name_Kristina Gutierrez', 'Agent_name_Madison Flores',
        'Agent_name_Richard Buchanan', 'Agent_name_Vicki Collins',
        
        # Manager (6)
        'Manager_Emily Chen', 'Manager_Jennifer Nguyen', 'Manager_John Smith',
        'Manager_Michael Lee', 'Manager_Olivia Tan', 'Manager_William Kim',
        
        # Agent Shift (5)
        'Agent_Shift_Afternoon', 'Agent_Shift_Evening', 'Agent_Shift_Morning',
        'Agent_Shift_Night', 'Agent_Shift_Split'
    ]

def preprocess_input(data_dict):
    """Preprocesses the input data with one-hot encoding"""
    df = pd.DataFrame([data_dict])
    
    # Define categorical columns to encode
    categorical_columns = [
        'channel_name', 'category', 'Sub-category', 'Agent_name',
        'Manager', 'Agent_Shift'
    ]
    
    # Create dummy variables
    df_encoded = pd.get_dummies(df[categorical_columns], columns=categorical_columns)
    
    # Debug information - only showing encoded column names
    st.write("Original encoded column names:", sorted(list(df_encoded.columns)))
    
    # Ensure all expected columns are present
    expected_columns = get_expected_columns()
    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Keep only expected columns in the correct order
    df_encoded = df_encoded[expected_columns]
    
    # Convert to tensor
    return tf.convert_to_tensor(df_encoded.values, dtype=tf.float32)

def main():
    st.title("Customer Satisfaction Prediction System")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            channel_name = st.selectbox(
                "Channel Name",
                ['Outcall', 'Inbound', 'Email']
            )
            
            category = st.selectbox(
                "Category",
                ['Product Queries', 'Order Related', 'Returns', 
                'Cancellation', 'Shopzilla Related', 'Payments related', 'Refund Related', 
                'Feedback', 'Offers & Cashback', 'Onboarding related', 'Others', 'App/website']
            )
            
            sub_category = st.selectbox(
                "Sub-category",
                ['Life Insurance', 'Product Specific Information', 
                'Installation/demo', 'Reverse Pickup Enquiry', 'Not Needed', 'Fraudulent User',
                'Exchange / Replacement', 'Missing', 'General Enquiry', 'other']
            )
        
        with col2:
            agent_name = st.selectbox(
                "Agent Name",
                ['Richard Buchanan', 'Vicki Collins', 'Duane Norman', 
                'Madison Flores', 'Brenda Gillespie', 'Kristina Gutierrez']
            )
            
            manager = st.selectbox(
                "Manager",
                ['Jennifer Nguyen', 'Michael Lee', 'William Kim', 
                'John Smith', 'Olivia Tan', 'Emily Chen']
            )
            
            agent_shift = st.selectbox(
                "Agent Shift",
                ["Morning", "Evening", "Split", "Afternoon", "Night"]
            )
        
        submit_button = st.form_submit_button("Predict CSAT Score")
    
    if submit_button:
        try:
            # Prepare input data
            input_data = {
                'channel_name': channel_name,
                'category': category,
                'Sub-category': sub_category,
                'Agent_name': agent_name,
                'Manager': manager,
                'Agent_Shift': agent_shift
            }
            
            # Preprocess input
            processed_input = preprocess_input(input_data)
            
            # Create and use model
            model = create_model()
            prediction = model.predict(processed_input)
            predicted_class = np.argmax(prediction[0])
            
            # Display prediction results
            st.markdown("### Prediction Results")
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = predicted_class + 1,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Predicted CSAT Score"},
                gauge = {
                    'axis': {'range': [1, 5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [1, 2], 'color': "red"},
                        {'range': [2, 3], 'color': "orange"},
                        {'range': [3, 4], 'color': "yellow"},
                        {'range': [4, 5], 'color': "green"}
                    ]
                }
            ))
            
            st.plotly_chart(fig)
            
            satisfaction_levels = [
                "Very Dissatisfied", "Dissatisfied", "Neutral",
                "Satisfied", "Very Satisfied"
            ]
            st.info(f"The customer is predicted to be: {satisfaction_levels[predicted_class]}")
            
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()