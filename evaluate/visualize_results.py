import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize(image, target, class_names=None):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    boxes = target['boxes']
    labels = target['labels']

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        if class_names:
            label_text = class_names[labels[i].item()]
            ax.text(x1, y1 - 5, label_text, color='yellow', fontsize=10, backgroundcolor='black')

    plt.axis("off")
    plt.show()
