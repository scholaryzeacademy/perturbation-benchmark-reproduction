from src.perturbation_conditions import perturbed_genes


def test_control_has_no_perturbed_genes():
    assert perturbed_genes("ctrl") == []


def test_single_gene_perturbation():
    assert perturbed_genes("GENE1+ctrl") == ["GENE1"]


def test_combinatorial_perturbation():
    assert perturbed_genes("GENE1+GENE2") == ["GENE1", "GENE2"]
