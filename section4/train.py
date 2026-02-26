import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. تعریف معماری شبکه عصبی بسیار ساده
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # یک لایه کانولوشن: 1 کانال ورودی (خاکستری)، 4 کانال خروجی (کرنل)، سایز 3x3
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.relu = nn.ReLU()
        # لایه پولینگ: سایز 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # لایه تمام متصل: 4 کانال * 13 * 13 پیکسل = 676 ورودی، 10 خروجی (کلاس های 0 تا 9)
        self.fc = nn.Linear(4 * 13 * 13, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 4 * 13 * 13) # تبدیل به آرایه یک بعدی (Flatten)
        x = self.fc(x)
        return x

# 2. آماده سازی داده ها و آموزش (فقط 1 ایپاک برای تست)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("درحال آموزش شبکه (حدود 1 دقیقه طول میکشد)...")
for images, labels in train_loader:
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
print("آموزش تمام شد!")

# 3. استخراج وزن ها به فرمت فایل هدر C
with open("weights.h", "w") as f:
    f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")
    
    # وزن های کانولوشن
    conv_w = model.conv1.weight.data.numpy()
    conv_b = model.conv1.bias.data.numpy()
    
    f.write("float conv_weights[4][3][3] = {\n")
    for out_c in range(4):
        f.write("  {\n")
        for i in range(3):
            f.write("    {" + ", ".join([f"{val}f" for val in conv_w[out_c, 0, i]]) + "},\n")
        f.write("  },\n")
    f.write("};\n\n")
    
    f.write("float conv_bias[4] = {" + ", ".join([f"{val}f" for val in conv_b]) + "};\n\n")

    # وزن های لایه تمام متصل
    fc_w = model.fc.weight.data.numpy()
    fc_b = model.fc.bias.data.numpy()
    
    f.write("float fc_weights[10][676] = {\n")
    for i in range(10):
        f.write("  {" + ", ".join([f"{val}f" for val in fc_w[i]]) + "},\n")
    f.write("};\n\n")
    
    f.write("float fc_bias[10] = {" + ", ".join([f"{val}f" for val in fc_b]) + "};\n\n")
    f.write("#endif\n")

print("فایل weights.h با موفقیت ساخته شد!")
