import json
import cv2
import pickle


def debug(st):
    print('*'*30, 'DEBUG:', st)

def save_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

# Load COCO annotations
def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# Save COCO annotations to a JSON file
def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def video_bucket(height, width, outpath, frame_rate=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outpath, fourcc, frame_rate, (width, height))
    return out

# Print basic information about the dataset (.json in coco format)
def print_dataset_info(coco_data):
    print('\n')
    print('*'*50)
    print("Categories:")
    for category in coco_data['categories']:
        print(f"  - ID: {category['id']}, Name: {category['name']}")
    
    num_images = len(coco_data['images'])
    print(f"\nNumber of images: {num_images}")
    
    num_annotations = len(coco_data['annotations'])
    print(f"Number of annotations: {num_annotations}")
    
    annotated_image_ids = {ann['image_id'] for ann in coco_data['annotations']}
    print(f"Number of unique images with annotations: {len(annotated_image_ids)}")
    
    print("\nAnnotation distribution per category:")
    category_counts = {cat['id']: 0 for cat in coco_data['categories']}
    for ann in coco_data['annotations']:
        category_counts[ann['category_id']] += 1
    for category in coco_data['categories']:
        print(f"  - {category['name']}: {category_counts[category['id']]} annotations")
    print('*'*50)
    
    print(f'Example Image: {coco_data["images"][0]}')
    print('\n')
    print(f'Example Annotation: {coco_data["annotations"][0]}')
    print('\n')


def print_model_info(model, model_name="Model", verbose=False):
    """
    Print information about a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze.
        model_name (str): The name of the model for display.
    """
    print(f"\nModel Name: {model_name}")
    print("=" * 40)

    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total     Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    if verbose:
        print("-" * 40)
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(name, p.numel())

    # Print a summary of the layers
    # print("{:<30} {:<15} {:<15}".format("Layer Name", "Output Shape", "Param #"))
    # print("-" * 60)
    
    # for name, module in model.named_modules():
    #     if len(list(module.parameters())) > 0:  # Only include layers with parameters
    #         params = sum(p.numel() for p in module.parameters())
    #         output_shape = "-"
    #         if hasattr(module, "out_features"):
    #             output_shape = f"({module.out_features})"
    #         elif hasattr(module, "out_channels"):
    #             output_shape = f"({module.out_channels}, ...)"
    #         print(f"{name:<30} {output_shape:<15} {params:<15,}")

    print("=" * 40)
    print('\n')
