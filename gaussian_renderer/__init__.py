from gaussian_renderer.render import render
from gaussian_renderer.neilf import render_neilf
from gaussian_renderer.vndf_sampling import render_vndf_sampling

render_fn_dict = {
    "render": render,
    "neilf": render_neilf,
    "vndf_sampling": render_vndf_sampling,
}
