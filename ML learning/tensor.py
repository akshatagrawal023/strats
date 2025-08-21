import torch

my_values = [[1,2,3], [2,3,4], [3,4,5], [4,5,6]]

my_tensor = torch.tensor(my_values)

print(my_tensor.shape)
print(my_tensor.device)
print("Is GPU Available? ", torch.backends.mps.is_available())

#moving tensor to GPU
gpu = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
my_tensor = my_tensor.to(gpu)

#slicing and Tensor Math
left_tensor = my_tensor[:2, :]
right_tensor = my_tensor[2:, :]
print(left_tensor)

print("1st way: ",left_tensor + right_tensor)
print("2nd way: ",left_tensor.add(right_tensor))

#Elementwise Multiplication
ew_tensor_operator = left_tensor * right_tensor
ew_tensor_method = left_tensor.mul(right_tensor)

print(ew_tensor_method, ew_tensor_operator)


#Matrix Multiplication
new_left_tensor = torch.Tensor([[2,5] , [7,3]])
new_right_tensor = torch.Tensor([[8],[9]])

print("1st way", new_left_tensor @ new_right_tensor)
print("2nd way ",new_left_tensor.matmul(new_right_tensor))

print(my_tensor.mean()) #gives mean of all the values in a tensor

#Mean for each column
print(my_tensor.mean(dim = [0])) #Dimention = 0 for row and 1 for column

