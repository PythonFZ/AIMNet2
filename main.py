from aimnet2 import AIMNet2
import zntrack

with zntrack.Project() as proj:
    b973c_0 = AIMNet2(model_path="models/aimnet2_b973c_0.jpt", name="b973c_0")
    b973c_1 = AIMNet2(model_path="models/aimnet2_b973c_1.jpt", name="b973c_1")
    b973c_2 = AIMNet2(model_path="models/aimnet2_b973c_2.jpt", name="b973c_2")
    b973c_3 = AIMNet2(model_path="models/aimnet2_b973c_3.jpt", name="b973c_3")
    b973c_ens = AIMNet2(model_path="models/aimnet2_b973c_ens.jpt", name="b973c_ens")

    wb97m_d3_0 = AIMNet2(model_path="models/aimnet2_wb97m-d3_0.jpt", name="wb97m_d3_0")
    wb97m_d3_1 = AIMNet2(model_path="models/aimnet2_wb97m-d3_1.jpt", name="wb97m_d3_1")
    wb97m_d3_2 = AIMNet2(model_path="models/aimnet2_wb97m-d3_2.jpt", name="wb97m_d3_2")
    wb97m_d3_3 = AIMNet2(model_path="models/aimnet2_wb97m-d3_3.jpt", name="wb97m_d3_3")
    wb97m_d3_ens = AIMNet2(
        model_path="models/aimnet2_wb97m-d3_ens.jpt", name="wb97m_d3_ens"
    )

proj.build()
