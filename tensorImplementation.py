class Tensor:
    def __init__(self, data):
        self.data = data
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

    
    
