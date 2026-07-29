# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F


# # class VisualImportanceModel(nn.Module):
# #     def __init__(self, input_dim=512, dropout=0.3):
# #         super().__init__()

# #         # -------- Layer 1 --------
# #         self.fc1 = nn.Linear(input_dim, 1024)
# #         self.bn1 = nn.BatchNorm1d(1024)

# #         # -------- Layer 2 --------
# #         self.fc2 = nn.Linear(1024, 512)
# #         self.bn2 = nn.BatchNorm1d(512)

# #         # -------- Residual Projection --------
# #         self.res = nn.Linear(input_dim, 512)

# #         # -------- Layer 3 --------
# #         self.fc3 = nn.Linear(512, 256)
# #         self.bn3 = nn.BatchNorm1d(256)

# #         # -------- Layer 4 --------
# #         self.fc4 = nn.Linear(256, 128)
# #         self.bn4 = nn.BatchNorm1d(128)

# #         # -------- Layer 5 --------
# #         self.fc5 = nn.Linear(128, 64)
# #         self.bn5 = nn.BatchNorm1d(64)

# #         # -------- Feature Gating --------
# #         self.gate = nn.Linear(64, 64)

# #         # -------- Output --------
# #         self.out = nn.Linear(64, 1)

# #         self.dropout = nn.Dropout(dropout)

# #     def forward(self, v):
# #         # Layer 1
# #         x = F.relu(self.bn1(self.fc1(v)))
# #         x = self.dropout(x)

# #         # Layer 2
# #         x = F.relu(self.bn2(self.fc2(x)))
# #         x = self.dropout(x)

# #         # Residual connection
# #         x = x + self.res(v)

# #         # Layer 3
# #         x = F.relu(self.bn3(self.fc3(x)))
# #         x = self.dropout(x)

# #         # Layer 4
# #         x = F.relu(self.bn4(self.fc4(x)))
# #         x = self.dropout(x)

# #         # Layer 5
# #         x = F.relu(self.bn5(self.fc5(x)))

# #         # Feature gating
# #         gate = torch.sigmoid(self.gate(x))
# #         x = x * gate

# #         # Output
# #         return torch.sigmoid(self.out(x)).squeeze(1)



import torch
import torch.nn as nn

class VisualImportanceModel(nn.Module):
    """
    Visual Importance Prediction Network
    Architecture: 512 → 256 → 128 → 64 → 1
    Based on research paper with batch normalization, dropout, and feature gating
    """
    def __init__(self, input_dim=512, dropout=0.4):
        super().__init__()
        
        # Layer 1: 512 → 256
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout)
        
        # Layer 2: 256 → 128
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout)
        
        # Layer 3: 128 → 64
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout)
        
        # Feature Gating (sigmoid mask)
        self.gate = nn.Linear(64, 64)
        
        # Output Layer: 64 → 1
        self.fc4 = nn.Linear(64, 1)

    def forward(self, v):
        # Layer 1
        x = self.fc1(v)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        
        # Feature gating (attention mechanism)
        gate = torch.sigmoid(self.gate(x))
        x = x * gate
        
        # Output with sigmoid for [0,1] importance score
        return torch.sigmoid(self.fc4(x)).squeeze(1)





# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class VisualImportanceModel(nn.Module):
#     def __init__(self, input_dim=512):
#         super().__init__()

#         # 512 -> 256
#         self.fc1 = nn.Linear(input_dim, 256)

#         # 256 -> 128
#         self.fc2 = nn.Linear(256, 128)

#         # 128 -> 64
#         self.fc3 = nn.Linear(128, 64)

#         # 64 -> 1
#         self.fc4 = nn.Linear(64, 1)

#     def forward(self, v):
#         x = F.relu(self.fc1(v))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = torch.sigmoid(self.fc4(x))

#         return x.squeeze(1)
