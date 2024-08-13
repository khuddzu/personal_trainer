import torch
import torchani

# from torchani.repulsion import StandaloneRepulsionCalculator

a0 = 0.529177249  # Bohr Radius


def ea02debye(x):
    return x / 0.3934303


def eA2debeye(x):
    return x / 0.20819434


"""
The below network utlizes the charges in the nmax charge correction method.
"""


class ANIModelCharge(torch.nn.ModuleList):
    """Units for MV Data:
    Charge: au
    Coordinate: A
    Energy: Ha
    Dipole: Debye
    """

    def __init__(self, modules, aev_computer):
        super(ANIModelCharge, self).__init__(modules)
        self.reducer = torch.sum
        self.padding_fill = 0
        self.aev_computer = aev_computer
        # self.repulsion = StandaloneRepulsionCalculator(elements=('H', 'C', 'N', 'O', 'S', 'F', 'Cl'))
        electronegativity = torch.tensor([7.18, 6.26, 7.27, 7.54, 6.22, 10.41, 8.29])
        chemical_hardness = torch.tensor(
            [12.84, 10.00, 14.53, 12.16, 8.28, 14.02, 9.35]
        )
        # nmax = (electronegativity**2)/(2*chemical_hardness)
        # #electrophilicity
        # .type(torch.LongTensor)       #nmax
        nmax = electronegativity / chemical_hardness
        self.register_buffer("nmax", nmax)

    def get_atom_mask(self, species):
        padding_mask = (species.ne(-1)).float()
        assert padding_mask.sum() > 1.0e-6
        padding_mask = padding_mask.unsqueeze(-1)
        return padding_mask

    def get_atom_neighbor_mask(self, atom_mask):
        atom_neighbor_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        assert atom_neighbor_mask.sum() > 1.0e-6
        return atom_neighbor_mask

    def get_coulomb(self, charges, coordinates, species):
        dist = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)
        # add 1e-6 to prevent sqrt(0) errors
        distances = torch.sqrt(torch.sum(dist**2, dim=-1) + 1e-6).unsqueeze(-1)
        # Mask for padding atoms in distance matrix.
        distance_matrix_mask = self.get_atom_neighbor_mask(self.get_atom_mask(species))
        charges = charges.unsqueeze(2)
        charge_products = charges.unsqueeze(1) * charges.unsqueeze(2)
        coulomb = charge_products / (distances)
        coulomb = coulomb * distance_matrix_mask
        coulomb = coulomb.squeeze(-1)
        coulomb = torch.triu(coulomb, diagonal=1)
        coulomb = torch.sum(coulomb, dim=(1, 2))
        return coulomb * a0  # Ha

    # Correction function for Nmax only
    # def get_correction(self, excess_charge, species):
    # nmax_ = self.nmax.unsqueeze(1)
    # nmax_matrix = nmax_[species]
    # new_nmax_matrix = nmax_matrix *  self.get_atom_mask(species)
    # nmax_sum = new_nmax_matrix.sum(dim=1)
    # charge_corrections = (new_nmax_matrix.squeeze(-1)/nmax_sum)*excess_charge
    # sign = 2*(excess_charge>0).to(torch.long)-1
    # charge_corrections = sign*charge_corrections
    # return charge_corrections

    # Correction function that implements nmax and the networks charge predictions
    def get_correction(self, excess_charge, species, charges):
        nmax_ = self.nmax.unsqueeze(1)
        charges = charges.unsqueeze(2)
        nmax_matrix = nmax_[species]
        new_nmax_matrix = nmax_matrix * self.get_atom_mask(species)
        qn_matrix = charges.squeeze(-1) * new_nmax_matrix.squeeze(-1)
        qn_sum = (qn_matrix**2).sum(dim=1)
        charge_corrections = ((qn_matrix) ** 2 / (qn_sum.unsqueeze(1))) * excess_charge
        return charge_corrections

    def forward(self, species_coordinates, total_charge=0):
        species, coordinates = species_coordinates
        species, aev = self.aev_computer(species_coordinates)
        species_ = species.flatten()
        num_atoms = (species.ne(-1)).sum(dim=1, dtype=aev.dtype)
        present_species = torchani.utils.present_species(species)
        aev = aev.flatten(0, 1)
        output = torch.full_like(species_, self.padding_fill, dtype=aev.dtype)
        output_c = torch.full_like(species_, self.padding_fill, dtype=aev.dtype)
        for i in present_species:
            # Check that none of the weights are nan.
            for parameter in self[i].parameters():
                assert not (torch.isnan(parameter)).any()
            mask = species_ == i
            input_ = aev.index_select(0, mask.nonzero().squeeze())
            res = self[i](input_)
            output.masked_scatter_(mask, res[:, 0].squeeze())
            output_c.masked_scatter_(mask, res[:, 0].squeeze())  # changed

        output = output.view_as(species)
        output_c = output_c.view_as(species)
        # Maintain conservation of charge
        excess_charge = torch.full_like(output_c[:, 0], total_charge) - torch.sum(
            output_c, dim=1
        )
        excess_charge = excess_charge.unsqueeze(1)
        correction = self.get_correction(excess_charge, species, output_c)
        output_c = (output_c + correction) * self.get_atom_mask(species).squeeze(-1)
        # rep_energy = self.repulsion((species, coordinates)).energies
        molecular_energies = self.reducer(output, dim=1)
        coulomb = self.get_coulomb(output_c, coordinates, species)
        molecular_energies = molecular_energies + coulomb  # + rep_energy
        # return species, molecular_energies, output
        return (
            species,
            molecular_energies,
            output,
            output_c,
            excess_charge,
            coulomb,
            correction,
        )
        # return species, output_c, excess_charge, correction
