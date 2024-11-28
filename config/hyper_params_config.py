#loss weight
text_contrastive_loss_w=0.2
audio_contrastive_loss_w=0.2
classification_loss_w=1
binary_classification_loss_w=0.2
modal_contrastive_loss_w=0.2
aug_classification_loss_w=1
aug_text_recon_loss_w=0.2
aug_audio_recon_loss_w=0.2

audio_kl_w=1
text_kl_w=1

# Instantiate models
num_labels = 4
batch_size=4
#bert hidden_size
hidden_size=1024
