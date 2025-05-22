def get_class_name(class_id, coco):
    return coco.loadCats(class_id)[0]["name"]

def filter_by_weather(coco, weather="foggy"):
    return [img for img in coco.dataset["images"] if img.get("attributes", {}).get("weather") == weather]
