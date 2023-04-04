import torch


def generator(pretrained=True, device="cpu", progress=True, check_hash=True):
    from model import Generator

    release_url = "https://github.com/bryandlee/animegan2-pytorch/raw/main/weights"
    release_url = "https://github.com/liujianzuo/animegan2-pytorch/raw/main/weights"
    known = {
        name: f"{release_url}/{name}.pt"
        for name in [
            'celeba_distill', 'face_paint_512_v1', 'face_paint_512_v2', 'paprika',
            'pytorch_generator_Hayao',  'pytorch_generator_Hayao_v2', 'pytorch_generator_Paprika','pytorch_generator_Shinkai',  # v2的tensor转换而来
            'generator_celeba_distill', 'face_paint_512_v2_0','face_paint_512_v0', # 原项目下载
            'Hayao_net_G_float', 'Hosoda_net_G_float','Paprika_net_G_float','Shinkai_net_G_float' # https://github.com/ahmedbesbes/cartoonify#fromHistory   这里下载的有个sh脚本

        ]
    }

    device = torch.device(device)
    model = Generator().to(device)

    if type(pretrained) == str:
        # Look if a known name is passed, otherwise assume it's a URL
        ckpt_url = known.get(pretrained, pretrained)
        pretrained = True
    else:
        ckpt_url = known.get('face_paint_512_v2')

    if pretrained is True:
        print(ckpt_url)
        state_dict = torch.hub.load_state_dict_from_url(
            ckpt_url,
            map_location=device,
            progress=progress,
            check_hash=check_hash,
        )
        model.load_state_dict(state_dict)

    return model


def face2paint(device="cpu", size=512, side_by_side=False):
    from PIL import Image
    from torchvision.transforms.functional import to_tensor, to_pil_image


    def face2paint(
        model: torch.nn.Module,
        img: Image.Image,
        size: int = size,
        side_by_side: bool = side_by_side,
        device: str = device,
    ) -> Image.Image:
        w, h = img.size

        bili = h/w  # 图片比例不变
        # print(w, h, size) # 不改图片尺寸

        s = min(w, h)
        # img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)) # 不裁剪
        if w < 700:
            suoxiao = 1
        else:
            suoxiao = 1  #缩小倍数

        img = img.resize((int(w/suoxiao), int((w/suoxiao)*bili)), Image.LANCZOS) #比例放大缩小

        with torch.no_grad():
            input = to_tensor(img).unsqueeze(0) * 2 - 1
            output = model(input.to(device)).cpu()[0]

            if side_by_side:
                output = torch.cat([input[0], output], dim=2)

            output = (output * 0.5 + 0.5).clip(0, 1)

        return to_pil_image(output)

    return face2paint
