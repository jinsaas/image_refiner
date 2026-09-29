
def apply_image_2cut(ref_canvas, image_a, image_b, mode="vertical", pad_color="#FFFFFF", resize_set="NEAREST"):
# Dedicated fixed-padding function group. Removed from the applied functions due to poor efficiency.

    H, W = ref_canvas.shape[1:3]
    Ha, Wa = image_a.shape[1:3]
    Hb, Wb = image_b.shape[1:3]


    # resize
    if mode == "vertical":
        image_a = resize_keep_ratio(image_a, target_w=W-32, resize_set=resize_set)
        image_b = resize_keep_ratio(image_b, target_w=W-32, resize_set=resize_set)
        Ha, Wa = image_a.shape[1:3]
        Hb, Wb = image_b.shape[1:3]
    else:  # horizontal
        image_a = resize_keep_ratio(image_a, target_h=H-32, resize_set=resize_set)
        image_b = resize_keep_ratio(image_b, target_h=H-32, resize_set=resize_set)
        Ha, Wa = image_a.shape[1:3]
        Hb, Wb = image_b.shape[1:3]

    # padding logic
    if mode == "vertical":
        Y = H-16
        padded_a = apply_padding(image_a, top=16, bottom=max(0, Y - Hb), left=16, right=16, pad_color=pad_color)
        padded_b = apply_padding(image_b, top=max(0, Y - Ha), bottom=16, left=16, right=16, pad_color=pad_color)
        return padded_a, padded_b

    elif mode == "horizontal":
        X = W-16
        padded_a = apply_padding(image_a, top=16, bottom=16, left=16, right=max(0, X - Wb), pad_color=pad_color)
        padded_b = apply_padding(image_b, top=16, bottom=16, left=max(0, X - Wa), right=16, pad_color=pad_color)
        return padded_a, padded_b

    else:
        return ref_canvas

def composite_fixed_2cut(ref_canvas, padded_a, padded_b, masks):
# Dedicated fixed-padding function group. Removed from the applied functions due to poor efficiency.

    mask_a = masks["A"].float().repeat(1,1,1,3)  # (1,H,W,3)
    mask_b = masks["B"].float().repeat(1,1,1,3)  # (1,H,W,3)
    composite_a = ref_canvas * (1 - mask_a) + padded_a * mask_a
    composite = composite_a * (1 - mask_b) + padded_b * mask_b
    return composite

def apply_image_3cut(ref_canvas, image_a, image_b, image_c, mode="vertical", pad_color="#FFFFFF", resize_set="NEAREST"):
# Dedicated fixed-padding function group. Removed from the applied functions due to poor efficiency.

    H, W = ref_canvas.shape[1:3]

    if mode == "vertical":
        image_a = resize_keep_ratio(image_a, target_w=W-32, resize_set=resize_set)
        image_b = resize_keep_ratio(image_b, target_w=W-32, resize_set=resize_set)
        image_c = resize_keep_ratio(image_c, target_w=W-32, resize_set=resize_set)
    else:
        image_a = resize_keep_ratio(image_a, target_h=H-32, resize_set=resize_set)
        image_b = resize_keep_ratio(image_b, target_h=H-32, resize_set=resize_set)
        image_c = resize_keep_ratio(image_c, target_h=H-32, resize_set=resize_set)

    if mode == "vertical":
        padded_a = apply_padding(image_a, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
        padded_b = apply_padding(image_b, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
        padded_c = apply_padding(image_c, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
        return padded_a, padded_b, padded_c
    else:
        padded_a = apply_padding(image_a, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
        padded_b = apply_padding(image_b, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
        padded_c = apply_padding(image_c, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
        return padded_a, padded_b, padded_c

def composite_fixed_3cut(ref_canvas, padded_a, padded_b, padded_c, masks):
# Dedicated fixed-padding function group. Removed from the applied functions due to poor efficiency.

    mask_a = masks["A"].float().repeat(1, 1, 1, 3)
    mask_b = masks["B"].float().repeat(1, 1, 1, 3)
    mask_c = masks["C"].float().repeat(1, 1, 1, 3)
    
    composite_a = ref_canvas * (1 - mask_a) + padded_a * mask_a
    composite_b = composite_a * (1 - mask_b) + padded_b * mask_b
    composite = composite_b * (1 - mask_c) + padded_c * mask_c
    return composite


def apply_image_4cut(ref_canvas, image_a, image_b, image_c, image_d, mode="vertical", pad_color="#FFFFFF", resize_set="NEAREST"):
# Dedicated fixed-padding function group. Removed from the applied functions due to poor efficiency.

    H, W = ref_canvas.shape[1:3]
    
    if mode == "vertical":
        image_a = resize_keep_ratio(image_a, target_w=W-32, resize_set=resize_set)
        image_b = resize_keep_ratio(image_b, target_w=W-32, resize_set=resize_set)
        image_c = resize_keep_ratio(image_c, target_w=W-32, resize_set=resize_set)
        image_d = resize_keep_ratio(image_d, target_w=W-32, resize_set=resize_set)
    else:
        image_a = resize_keep_ratio(image_a, target_h=H-32, resize_set=resize_set)
        image_b = resize_keep_ratio(image_b, target_h=H-32, resize_set=resize_set)
        image_c = resize_keep_ratio(image_c, target_h=H-32, resize_set=resize_set)
        image_d = resize_keep_ratio(image_d, target_h=H-32, resize_set=resize_set)

    padded_a = apply_padding(image_a, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
    padded_b = apply_padding(image_b, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
    padded_c = apply_padding(image_c, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
    padded_d = apply_padding(image_d, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
    
    return padded_a, padded_b, padded_c, padded_d

def composite_fixed_4cut(ref_canvas, padded_a, padded_b, padded_c, padded_d, masks):
# Dedicated fixed-padding function group. Removed from the applied functions due to poor efficiency.

    mask_a = masks["A"].float().repeat(1, 1, 1, 3)
    mask_b = masks["B"].float().repeat(1, 1, 1, 3)
    mask_c = masks["C"].float().repeat(1, 1, 1, 3)
    mask_d = masks["D"].float().repeat(1, 1, 1, 3)
    
    composite = ref_canvas * (1 - mask_a) + padded_a * mask_a
    composite = composite * (1 - mask_b) + padded_b * mask_b
    composite = composite * (1 - mask_c) + padded_c * mask_c
    composite = composite * (1 - mask_d) + padded_d * mask_d
    return composite

def apply_image_5cut(ref_canvas, image_a, image_b, image_c, image_d, image_e, mode="vertical", pad_color="#FFFFFF", resize_set="NEAREST"):
# Dedicated fixed-padding function group. Removed from the applied functions due to poor efficiency.

    H, W = ref_canvas.shape[1:3]
    
    imgs = [image_a, image_b, image_c, image_d, image_e]
    resized = []
    for img in imgs:
        if mode == "vertical":
            resized.append(resize_keep_ratio(img, target_w=W-32, resize_set=resize_set))
        else:
            resized.append(resize_keep_ratio(img, target_h=H-32, resize_set=resize_set))

    padded = [apply_padding(img, top=16, bottom=16, left=16, right=16, pad_color=pad_color) for img in resized]
    return tuple(padded)

def composite_fixed_5cut(ref_canvas, padded_a, padded_b, padded_c, padded_d, padded_e, masks):
# Dedicated fixed-padding function group. Removed from the applied functions due to poor efficiency.

    mask_a = masks["A"].float().repeat(1, 1, 1, 3)
    mask_b = masks["B"].float().repeat(1, 1, 1, 3)
    mask_c = masks["C"].float().repeat(1, 1, 1, 3)
    mask_d = masks["D"].float().repeat(1, 1, 1, 3)
    mask_e = masks["E"].float().repeat(1, 1, 1, 3)
    
    composite = ref_canvas * (1 - mask_a) + padded_a * mask_a
    composite = composite * (1 - mask_b) + padded_b * mask_b
    composite = composite * (1 - mask_c) + padded_c * mask_c
    composite = composite * (1 - mask_d) + padded_d * mask_d
    composite = composite * (1 - mask_e) + padded_e * mask_e
    return composite

def apply_image_6cut(ref_canvas, image_a, image_b, image_c, image_d, image_e, image_f, mode="vertical", pad_color="#FFFFFF", resize_set="NEAREST"):
# Dedicated fixed-padding function group. Removed from the applied functions due to poor efficiency.

    H, W = ref_canvas.shape[1:3]
    
    imgs = [image_a, image_b, image_c, image_d, image_e, image_f]
    resized = []
    for img in imgs:
        if mode == "vertical":
            resized.append(resize_keep_ratio(img, target_w=W-32, resize_set=resize_set))
        else:
            resized.append(resize_keep_ratio(img, target_h=H-32, resize_set=resize_set))

    padded = [apply_padding(img, top=16, bottom=16, left=16, right=16, pad_color=pad_color) for img in resized]
    return tuple(padded)

def composite_fixed_6cut(ref_canvas, padded_a, padded_b, padded_c, padded_d, padded_e, padded_f, masks):
# Dedicated fixed-padding function group. Removed from the applied functions due to poor efficiency.

    keys = ["A", "B", "C", "D", "E", "F"]
    paddings = [padded_a, padded_b, padded_c, padded_d, padded_e, padded_f]
    
    composite = ref_canvas.clone()
    for key, pad_img in zip(keys, paddings):
        if key in masks:
            m = masks[key].float().repeat(1, 1, 1, 3)
            composite = composite * (1 - m) + pad_img * m
    return composite