import torch
import torch.nn.functional as F


# def calculate_FAR_FRR(threshold, model, dataloader):
#     far_count = 0  # Counter for false acceptance
#     frr_count = 0  # Counter for false rejection
#     impostor_count = 0  # Counter for impostor triplets
#     genuine_count = 0  # Counter for genuine triplets
#
#     model.eval()
#
#     with torch.no_grad():
#         for batch in dataloader:
#             anchor, positive, negative = batch
#
#             # Pass samples through the model to get embeddings
#             anchor_embed, positive_embed, negative_embed = model(anchor, positive, negative)
#
#             # Compute distances
#             anchor_positive_distance = F.pairwise_distance(anchor_embed, positive_embed, p=2)
#             anchor_negative_distance = F.pairwise_distance(anchor_embed, negative_embed, p=2)
#
#             # Convert distances to binary predictions
#             anchor_positive_prediction = (
#                     anchor_positive_distance <= threshold).float()  # 1 if distance <= threshold (genuine), 0 otherwise
#             anchor_negative_prediction = (
#                     anchor_negative_distance <= threshold).float()  # 1 if distance <= threshold (genuine), 0 otherwise
#
#             # Update counts
#             genuine_count += torch.sum(anchor_positive_prediction).item()
#             impostor_count += (anchor_positive_prediction.shape[0] - torch.sum(anchor_positive_prediction)).item()
#
#             frr_count += (anchor_positive_prediction.shape[0] - torch.sum(anchor_positive_prediction)).item()
#             far_count += torch.sum(anchor_negative_prediction).item()
#
#     far = (far_count / impostor_count) * 100
#     frr = (frr_count / genuine_count) * 100
#
#     return far, frr
#
#
# def calculate_accuracy(threshold, model, dataloader):
#     correct = 0
#     total = 0
#
#     model.eval()
#
#     with torch.no_grad():
#         for batch in dataloader:
#             anchor, positive, negative = batch
#
#             # Pass samples through the model to get embeddings
#             anchor_embed, positive_embed, negative_embed = model(anchor, positive, negative)
#
#             # Compute distances
#             anchor_positive_distance = F.pairwise_distance(anchor_embed, positive_embed, p=2)
#
#             # Convert distances to binary predictions
#             predictions = (
#                     anchor_positive_distance <= threshold).float()  # 1 if distance <= threshold (genuine), 0 otherwise
#
#             # Update correct and total counts
#             # ground truth labels are in the third element of the batch tuple
#             correct += torch.eq(predictions, negative).all(dim=1).sum().item()
#             total += len(anchor)
#
#     accuracy = correct / total
#
#     return accuracy * 100

def calculate_FAR_FRR(threshold, model, dataloader):
    # FAR = (False Positive / (False Positive + True Negative)) * 100
    # FRR = (False Negative / (False Negative + True Positive)) * 100

    far_count = 0  # Counter for false acceptance
    frr_count = 0  # Counter for false rejection
    impostor_count = 0  # Counter for impostor pairs
    genuine_count = 0  # Counter for genuine pairs

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            image1, image2, label = batch

            # Pass samples through the model to get embeddings
            first_emb, second_emb = model(image1, image2)

            # Compute distances
            distances = F.pairwise_distance(first_emb, second_emb, p=2)

            # Convert distances to binary predictions
            predictions = (distances <= threshold).float()  # 1 if distance <= threshold (genuine), 0 otherwise

            # Update counts
            for i in range(len(label)):
                if label[i] == 1:  # Genuine pair
                    genuine_count += 1
                    if predictions[i] == 0:  # Incorrectly rejected
                        frr_count += 1
                else:  # Impostor pair
                    impostor_count += 1
                    if predictions[i] == 1:  # Incorrectly accepted
                        far_count += 1

    far = (far_count / impostor_count) * 100
    frr = (frr_count / genuine_count) * 100

    return far, frr


def calculate_accuracy(threshold, model, dataloader):
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            image1, image2, label = batch

            # Pass samples through the model to get embeddings
            first_emb, second_emb = model(image1, image2)

            # Compute distances
            distances = F.pairwise_distance(first_emb, second_emb, p=2)

            # Convert distances to binary predictions
            predictions = (distances <= threshold).float()  # 1 if distance <= threshold (genuine), 0 otherwise

            # Update correct and total counts
            # ground truth labels are in the third element of the batch tuple
            correct += torch.eq(predictions, label).all(dim=1).sum().item()
            total += len(image1)

    accuracy = correct / total

    return accuracy * 100
