from diffusion_utilities import *


class ContextUnet(nn.Module):
    def __init__(self, in_channels, n_feat=64, n_cfeat=5, height=64):  # cfeat - context features
        super(ContextUnet, self).__init__()

        # number of input channels, number of intermediate feature maps and number of classes
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.h = height  #assume h == w. must be divisible by 4, so 28,24,20,16...

        # Initialize the initial convolutional layer
        self.init_conv = ResidualConvBlock(in_channels, n_feat, is_res=True)
        # init_ #[b, 64, 64, 64]

        # Initialize the down-sampling path of the U-Net with two levels
        self.down1 = UnetDown(n_feat, n_feat)  # down1 #[b,  64, 32, 32]
        self.down2 = UnetDown(n_feat, 2 * n_feat)  # down2 #[b, 128, 16, 16]
        self.down3 = UnetDown(2 * n_feat, 4 * n_feat)  # down3 #[b, 256,  8,  8]
        self.down4 = UnetDown(4 * n_feat, 8 * n_feat)  # down4 #[b, 512,  4,  4]

        # original: self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())
        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.GELU())
        # to_ve #[b, 512,  1,  1]

        # Embed the timestep and context labels with a one-layer fully connected neural network
        self.timeembed1 = EmbedFC(1, 8 * n_feat)
        self.timeembed2 = EmbedFC(1, 4 * n_feat)
        self.timeembed3 = EmbedFC(1, 2 * n_feat)
        self.timeembed4 = EmbedFC(1, 1 * n_feat)
        self.contextembed1 = EmbedFC(n_cfeat, 8 * n_feat)
        self.contextembed2 = EmbedFC(n_cfeat, 4 * n_feat)
        self.contextembed3 = EmbedFC(n_cfeat, 2 * n_feat)
        self.contextembed4 = EmbedFC(n_cfeat, 1 * n_feat)

        # Initialize the up-sampling path of the U-Net with four levels
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(8 * n_feat, 8 * n_feat, 4, 4),  # alterei!! h//4 para 4
            nn.GroupNorm(8, 8 * n_feat),  # normalize
            nn.ReLU(),
        )
        #up0 [b, 512,  1,  1]
        self.up1 = UnetUp(16 * n_feat, 4 * n_feat)  #[b, 256,  4,  4]
        self.up2 = UnetUp(8 * n_feat, 2 * n_feat)  #[b, 128,  8,  8]
        self.up3 = UnetUp(4 * n_feat, 1 * n_feat)  #[b,  64, 16, 16]
        self.up4 = UnetUp(2 * n_feat, 1 * n_feat)  #[b,  32, 32, 32]

        # Initialize the final convolutional layers to map to the same number of channels as the input image
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            # reduce number of feature maps   #in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.GroupNorm(8, n_feat),  # normalize
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),  # map to same number of channels as input
        )

    def forward(self, x, t, c=None):
        """
        x : (batch, n_feat, h, w) : input image
        t : (batch, n_cfeat)      : time step
        c : (batch, n_classes)    : context label
        """
        # x is the input image, c is the context label, t is the timestep, context_mask says which samples to block the context on

        # pass the input image through the initial convolutional layer
        x = self.init_conv(x)
        # pass the result through the down-sampling path
        down1 = self.down1(x)  #[b,  64, 32, 32]
        down2 = self.down2(down1)  #[b, 128, 16, 16]
        down3 = self.down3(down2)  #[b, 256,  8,  8]
        down4 = self.down4(down3)  #[b, 512,  4,  4]
        # print("down1.shape", down1.shape)
        # print("down2.shape", down2.shape)
        # print("down3.shape", down3.shape)
        # print("down4.shape", down4.shape)

        # convert the feature maps to a vector and apply an activation
        hiddenvec = self.to_vec(down4)
        # print("hiddenvec.shape", hiddenvec.shape)
        #[b, 128,  1,  1]

        # mask out context if context_mask == 1
        if c is None:
            c = torch.zeros(x.shape[0], self.n_cfeat).to(x)

        # embed context and timestep
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 8, 1, 1)  # (batch, 2*n_feat, 1,1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 8, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat * 4, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat * 4, 1, 1)
        cemb3 = self.contextembed3(c).view(-1, self.n_feat * 2, 1, 1)
        temb3 = self.timeembed3(t).view(-1, self.n_feat * 2, 1, 1)
        cemb4 = self.contextembed4(c).view(-1, self.n_feat, 1, 1)
        temb4 = self.timeembed4(t).view(-1, self.n_feat, 1, 1)
        #print(f"uunet forward: cemb1 {cemb1.shape}. temb1 {temb1.shape}, cemb2 {cemb2.shape}. temb2 {temb2.shape}")

        # print("bf up0")
        #[b, 128,  4,  4]
        up1 = self.up0(hiddenvec)
        # print("up1.shape", up1.shape)
        # print("(cemb1*up1 + temb1).shape", (cemb1*up1 + temb1).shape)
        up2 = self.up1(cemb1 * up1 + temb1, down4)
        # print("up2.shape", up2.shape)
        up3 = self.up2(cemb2 * up2 + temb2, down3)
        # print("up3.shape", up3.shape)
        up4 = self.up3(cemb3 * up3 + temb3, down2)
        # print("up4.shape", up4.shape)
        up5 = self.up4(cemb4 * up4 + temb4, down1)
        # print("up5.shape", up5.shape)
        out = self.out(torch.cat((up5, x), 1))
        # print("out.shape", out.shape)
        return out
