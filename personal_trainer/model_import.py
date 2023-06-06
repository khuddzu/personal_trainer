import torch
def model_arc(aev_dim, out_size, bias=False, activation='GELU'):
    if bias == False:
        if activation == 'CELU':
            H_network = torch.nn.Sequential(
                torch.nn.Linear(aev_dim, 256, bias=False),
                torch.nn.CELU(0.1),
                torch.nn.Linear(256, 192, bias=False),
                torch.nn.CELU(0.1),
                torch.nn.Linear(192, 160, bias=False),
                torch.nn.CELU(0.1),
                torch.nn.Linear(160, out_size, bias=False)
            )


            C_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 224, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(224, 192, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(192, 160, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(160, out_size, bias=False)
            )


            N_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 192, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(192, 160, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(160, 128, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(128, out_size, bias=False)
            )
    

            O_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 192, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(192, 160, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(160, 128, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(128, out_size, bias=False)
            )

            S_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 160, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(160, 128, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(128, 96, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(96, out_size, bias=False)
            )

            F_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 160, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(160, 128, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(128, 96, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(96, out_size, bias=False)
            )


            Cl_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 160, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(160, 128, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(128, 96, bias=False),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(96, out_size, bias=False)
            )
        elif activation == 'GELU':
            H_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 256, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(256, 192, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(192, 160, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(160, out_size, bias=False)
            )


            C_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 224, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(224, 192, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(192, 160, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(160, out_size, bias=False)
            )
    

            N_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 192, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(192, 160, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(160, 128, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, out_size, bias=False)
            )


            O_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 192, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(192, 160, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(160, 128, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, out_size, bias=False)
            )

            S_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 160, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(160, 128, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, 96, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(96, out_size, bias=False)
            )

            F_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 160, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(160, 128, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, 96, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(96, out_size, bias=False)
            )


            Cl_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 160, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(160, 128, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, 96, bias=False),
                    torch.nn.GELU(),
                    torch.nn.Linear(96, out_size, bias=False)
            )
    elif bias == True:
        if activation == 'CELU':
            H_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 256),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(256, 192),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(192, 160),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(160, out_size)
            )


            C_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 224),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(224, 192),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(192, 160),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(160, out_size)
            )
    

            N_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 192),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(192, 160),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(160, 128),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(128, out_size)
            )


            O_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 192),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(192, 160),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(160, 128),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(128, out_size)
            )
            S_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 160),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(160, 128),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(128, 96),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(96, out_size)
            )


            F_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 160),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(160, 128),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(128, 96),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(96, out_size)
            )


            Cl_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 160),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(160, 128),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(128, 96),
                    torch.nn.CELU(0.1),
                    torch.nn.Linear(96, out_size)
            )
        elif activation =='GELU':
            H_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 256),
                    torch.nn.GELU(),
                    torch.nn.Linear(256, 192),
                    torch.nn.GELU(),
                    torch.nn.Linear(192, 160),
                    torch.nn.GELU(),
                    torch.nn.Linear(160, out_size)
            )


            C_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 224),
                    torch.nn.GELU(),
                    torch.nn.Linear(224, 192),
                    torch.nn.GELU(),
                    torch.nn.Linear(192, 160),
                    torch.nn.GELU(),
                    torch.nn.Linear(160, out_size)
            )


            N_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 192),
                    torch.nn.GELU(),
                    torch.nn.Linear(192, 160),
                    torch.nn.GELU(),
                    torch.nn.Linear(160, 128),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, out_size)
            )


            O_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 192),
                    torch.nn.GELU(),
                    torch.nn.Linear(192, 160),
                    torch.nn.GELU(),
                    torch.nn.Linear(160, 128),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, out_size)
            )
            S_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 160),
                    torch.nn.GELU(),
                    torch.nn.Linear(160, 128),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, 96),
                    torch.nn.GELU(),
                    torch.nn.Linear(96, out_size)
            )


            F_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 160),
                    torch.nn.GELU(),
                    torch.nn.Linear(160, 128),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, 96),
                    torch.nn.GELU(),
                    torch.nn.Linear(96, out_size)
            )


            Cl_network = torch.nn.Sequential(
                    torch.nn.Linear(aev_dim, 160),
                    torch.nn.GELU(),
                    torch.nn.Linear(160, 128),
                    torch.nn.GELU(),
                    torch.nn.Linear(128, 96),
                    torch.nn.GELU(),
                    torch.nn.Linear(96, out_size)
            )
    return H_network, C_network, N_network, O_network, S_network, F_network, Cl_network
