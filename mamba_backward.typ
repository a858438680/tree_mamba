= Mamba forward

$
Delta t_1 &= Delta t + b_t \
Delta t_2 &= "softplus"(Delta t_1) \
Delta A_(1 i) &= Delta t_2 A_i \
Delta A_i &= exp(Delta A_(1 i)) \
Delta B_i &= Delta t_2 B_i \
S_(m i) &= w_i S_(l i) + (1 - w_i) S_(r i) \
S_i &= Delta A_i S_(m i) + Delta B_i x \
y_1 &= sum_i C_i S_i + D x \
z_1 &= "silu"(z) \
y &= y_1 z_1
$

= Mamba backward

$
nabla_(y_1) &= nabla_y z_1 \
nabla_(z_1) &= nabla_y y_1 \
nabla_(z) &= nabla_(z_1) sigma(z) (1 + z (1 - sigma(z))) \
nabla_(C_i) &= nabla_(y_1) S_i \
nabla_(S_i) &= nabla_(y_1) C_i \
nabla_(D) &= nabla_(y_1) x \
nabla_(Delta A_i) &= nabla_(S_i) S_(m i) \
nabla_(S_(m i)) &= nabla_(S_i) Delta A_i \
nabla_(Delta B_i) &= nabla_(S_i) x \
nabla_(x) &= sum_i nabla_(S_i) Delta B_i + nabla_(y_1) D \
nabla_(w_i) &= nabla_(S_(m i)) (S_(l i) - S_(r i)) \
nabla_(S_(l i)) & = nabla_(S_(m i)) w_i \
nabla_(S_(r i)) & = nabla_(S_(m i)) (1 - w_i) \
nabla_(B_i) &= nabla_(Delta B_i) Delta t_2 \
nabla_(Delta A_(1 i)) &= nabla_(Delta A_i) Delta A_i \
nabla_(Delta t_2) &= sum_i nabla_(Delta A_(1 i)) A_i + nabla_(Delta B_i) B_i \
nabla_(A_i) &= nabla_(Delta A_(1 i)) Delta t_2 \
nabla_(Delta t_1) &= nabla_(Delta t_2) exp(Delta t_1) / (1 + exp(Delta t_1)) \
nabla_(Delta t) &= nabla_(Delta t_1) \
nabla_(b_t) &= nabla_(Delta t_1) \
$