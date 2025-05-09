import numpy as np

class SimpleRNN:
    """
    A simple Recurrent Neural Network implementation.
    
    This RNN processes sequential inputs by maintaining a hidden state that gets
    updated at each time step, allowing the network to "remember" information
    from previous steps.
    """
    def __init__(self, input_size=9, hidden_size=4, output_size=3):
        """
        Initialize the RNN with weights and biases.
        
        Args:
            input_size: Dimension of input features at each time step
            hidden_size: Dimension of hidden state
            output_size: Dimension of output at each time step
        """
        # Weight matrix for transforming input -> hidden (shape: hidden_size x input_size)
        self.W_xh = np.random.rand(hidden_size, input_size)
        # Bias for input -> hidden transformation (shape: hidden_size x 1)
        self.b_xh = np.zeros((hidden_size, 1))
        
        # Weight matrix for transforming hidden -> output (shape: output_size x hidden_size)
        self.W_ho = np.random.rand(output_size, hidden_size)
        # Bias for hidden -> output transformation (shape: output_size x 1)
        self.b_ho = np.zeros((output_size, 1))
        
        # Weight matrix for hidden -> hidden recurrent connection (shape: hidden_size x hidden_size)
        self.W_hh = np.random.rand(hidden_size, hidden_size)
        # Bias for hidden -> hidden transformation (shape: hidden_size x 1)
        self.b_hh = np.zeros((hidden_size, 1))
        
        # Store dimensions for debugging and visualization
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
    
    def forward(self, inputs):
        """
        Process a sequence of inputs and produce outputs at each time step.
        
        Args:
            inputs: List of input vectors, one for each time step
            
        Returns:
            steps_output: Dictionary mapping time steps to output vectors
            hidden_states: Dictionary mapping time steps to hidden state vectors
        """
        # Dictionaries to store outputs and hidden states at each time step
        steps_output = {}
        hidden_states = {}
        
        # Initialize hidden state to zeros at t=-1 (before first input)
        hidden_states[-1] = np.zeros((self.hidden_size, 1))
        
        print(f"Starting forward pass with {len(inputs)} time steps")
        
        # Process each input in the sequence
        for t in range(len(inputs)):
            print(f"\n--- Time step {t} ---")
            
            # Reshape input to column vector (shape: input_size x 1)
            x = np.array(inputs[t]).reshape(-1, 1)
            print(f"Input shape: {x.shape}")
            
            # Calculate input contribution to hidden state
            # W_xh: (hidden_size x input_size) @ x: (input_size x 1) -> (hidden_size x 1)
            input_contribution = np.dot(self.W_xh, x) + self.b_xh
            print(f"Input contribution shape: {input_contribution.shape}")
            
            # Calculate recurrent contribution from previous hidden state
            # W_hh: (hidden_size x hidden_size) @ h_prev: (hidden_size x 1) -> (hidden_size x 1)
            recurrent_contribution = np.dot(self.W_hh, hidden_states[t-1]) + self.b_hh
            print(f"Recurrent contribution shape: {recurrent_contribution.shape}")
            
            # Combine input and recurrent contributions to get current hidden state
            current_hidden = input_contribution + recurrent_contribution
            
            # Apply activation function (tanh) to get final hidden state
            hidden_states[t] = np.tanh(current_hidden)
            print(f"Hidden state shape: {hidden_states[t].shape}")
            
            # Calculate output from current hidden state
            # W_ho: (output_size x hidden_size) @ h_t: (hidden_size x 1) -> (output_size x 1)
            steps_output[t] = np.dot(self.W_ho, hidden_states[t]) + self.b_ho
            print(f"Output shape: {steps_output[t].shape}")
            
            # Print a sample of values for better understanding
            print(f"Sample hidden state values: {hidden_states[t][:2].flatten()}")
            print(f"Sample output values: {steps_output[t][:2].flatten()}")
        
        print("\nForward pass complete!")
        return steps_output, hidden_states


if __name__ == '__main__':
    # Network configuration
    sequence_length = 10  # Number of time steps in sequence
    input_size = 9        # Dimension of input at each time step
    hidden_size = 4       # Dimension of hidden state
    output_size = 3       # Dimension of output at each time step
    
    print(f"Creating RNN with: input_size={input_size}, hidden_size={hidden_size}, output_size={output_size}")
    
    # Initialize RNN model
    rnn = SimpleRNN(input_size, hidden_size, output_size)
    
    # Generate random input sequence
    # Each item in the list represents one time step's input vector
    inputs = [np.random.randn(input_size) for _ in range(sequence_length)]
    print(f"Generated random input sequence with {len(inputs)} time steps")
    
    # Run forward pass to get outputs and hidden states
    print("\nRunning forward pass...")
    outputs, hidden_states = rnn.forward(inputs)
    
    print(f"\nNetwork produced outputs at time steps: {list(outputs.keys())}")
    print(f"Network produced hidden states at time steps: {list(set(hidden_states.keys()) - {-1})}")
    
    # Print shapes of all components to understand the dimensions
    print("\nShape summary:")
    print(f"W_xh (hidden x input): {rnn.W_xh.shape}")
    print(f"W_hh (hidden x hidden): {rnn.W_hh.shape}")
    print(f"W_ho (output x hidden): {rnn.W_ho.shape}")
    print(f"Final hidden state: {hidden_states[sequence_length-1].shape}")
    print(f"Final output: {outputs[sequence_length-1].shape}")
