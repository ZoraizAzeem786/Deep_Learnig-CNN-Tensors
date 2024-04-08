class Tensor:
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return str(self.data)
    
    def shape(self):
        if self.data == []:
            return ()
        elif isinstance(self.data, int):
            return (1,)
        shape = [len(self.data)]
        temp = self.data[0]

        while isinstance(temp, list):
            shape.append(len(temp))
            # Check if all rows have the same length
            if any(len(row) != len(temp) for row in self.data):
                return "Invalid tensor shape"
            temp = temp[0]

        if len(shape) > 1:
            return tuple(shape)
        else:
            return (shape[0],)
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            raise TypeError
        
        self_data = self.data
        other_data = other.data

        self_shape = self.shape()
        other_shape = other.shape()

        if self_shape != other_shape:
            broadcast_len = max(len(self_shape), len(other_shape))

            self_shape = (1,)*(broadcast_len-len(self_shape)) + self_shape
            other_shape = (1,)*(broadcast_len-len(other_shape)) + other_shape

            # Check if shapes are compatible for broadcasting
            for size_self, size_other in zip(self_shape, other_shape):
                if size_self != size_other and size_self != 1 and size_other != 1:
                    raise ValueError("Tensors must have compatible shapes for broadcasting")
                
            # Modify tensors to match broadcast shape
            self_data = self._broadcast(self_data, self_shape)
            other_data = other._broadcast(other_data, other_shape)

        # Function to perform element-wise addition
        def add_arrays(arr1, arr2):
            # Initialize result array
            result = []

            # Iterate over the arrays and add corresponding elements
            for row1, row2 in zip(arr1, arr2):
                if isinstance(row1[0], list):
                    # If the rows contain nested lists, recurse
                    result.append(add_arrays(row1, row2))
                else:
                    # If the rows contain integers, perform element-wise addition
                    result.append([elem1 + elem2 for elem1, elem2 in zip(row1, row2)])
            return result
        
        result_data = add_arrays(self_data, other_data)


        return Tensor(result_data)

    
    def _broadcast(self, data, target_shape):
        if isinstance(data, (int, float)):
            return [[data] * target_shape[1]] * target_shape[0]
        else:
            return [
                [elem for elem in row] * target_shape[1]
                for row in data
            ]
            
    
    
