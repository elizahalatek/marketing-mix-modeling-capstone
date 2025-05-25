import matplotlib.pyplot as plt

def plot_roi(roi_df):
    roi_df = roi_df.sort_values('roi', ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(roi_df['channel'], roi_df['roi'])
    plt.ylabel('ROI')
    plt.title('Channel-wise ROI')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
