"""Single-rounding projected linear layer, saving only the native bf16 input."""
import torch

def product(x,w):
    if x.is_cuda and x.dtype==w.dtype==torch.bfloat16:
        return torch.mm(x,w.T,out_dtype=torch.float32)
    return x.float()@w.float().T

def projected(g,u,v):
    if u.shape[1]:g=g-u@(u.T@g)
    if v.shape[1]:g=g-(g@v)@v.T
    return g

class BlockedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,u,v,p0,r0):
        ctx.save_for_backward(x,w,u,v,p0,r0)
        y=product(x,w)
        if u.shape[1] or v.shape[1]:
            wf=w.float();dp=u.T@wf-p0;dr=wf@v-r0
            q=dr-u@(u.T@dr)
            y=y-(x.float()@dp.T)@u.T-(x.float()@v)@q.T
        return y

    @staticmethod
    def backward(ctx,g):
        x,w,u,v,p0,r0=ctx.saved_tensors;gf=g.float()
        dx=dw=None
        if ctx.needs_input_grad[0]:
            dx=gf@w.float()
            if u.shape[1] or v.shape[1]:
                dp=u.T@w.float()-p0;dr=w.float()@v-r0;q=dr-u@(u.T@dr)
                dx=dx-(gf@u)@dp-(gf@q)@v.T
            dx=dx.to(x.dtype)
        if ctx.needs_input_grad[1]:dw=projected(gf.T@x.float(),u,v).to(w.dtype)
        return dx,dw,None,None,None,None

def append_span(old,new):
    """Keep the existing span and add only independent directions, in float64."""
    if not new.numel():return old
    basis=old.double();candidates=new.double()
    # Repeated projection avoids rank growth from float32 residual noise.
    for col in candidates.T:
        for _ in range(2):
            if basis.shape[1]:col=col-basis@(basis.T@col)
        size=col.norm()
        if size>1e-5:basis=torch.cat([basis,(col/size)[:,None]],1)
    if basis.shape[1]:assert float((basis.T@basis-torch.eye(basis.shape[1],device=basis.device)).abs().max())<1e-5
    return basis.float()

class BlockSet:
    def __init__(self,model,base,names):
        assert not torch.backends.cuda.matmul.allow_tf32
        self.model,self.base,self.names=model,base,list(names)
        self.spans={};self.items={};self.active=False
        for name in self.names:
            m=model.get_submodule(name.removesuffix('.weight'));device=m.weight.device
            u=torch.empty(m.weight.shape[0],0,device=device);v=torch.empty(m.weight.shape[1],0,device=device)
            self.spans[name]=(u,v)
        self.rebuild();self.install()

    @torch.no_grad()
    def rebuild(self):
        for name,(u,v) in self.spans.items():
            m=self.model.get_submodule(name.removesuffix('.weight'))
            if u.shape[1] or v.shape[1]:
                wf=self.base[name].to(m.weight.device).float();p0=u.T@wf;r0=wf@v
            else:
                p0=u.T;r0=v
            self.items[name]=(m,u,v,p0,r0)

    def install(self):
        if self.active:raise RuntimeError('Block is already installed')
        for name in self.names:
            m=self.items[name][0]
            def forward(x,name=name):
                module,u,v,p0,r0=self.items[name];flat=x.reshape(-1,x.shape[-1])
                y=BlockedLinear.apply(flat,module.weight,u,v,p0,r0)
                if module.bias is not None:y=y+module.bias.float()
                return y.to(x.dtype).reshape(*x.shape[:-1],y.shape[-1])
            m.forward=forward
        self.active=True

    def close(self):
        if self.active:
            for name in self.names:del self.items[name][0].forward
        self.active=False

    @torch.no_grad()
    def add(self,factors,ids,kmax=32):
        by={}
        for gid in ids:by.setdefault(self.names[gid//kmax],[]).append(gid%kmax)
        for name,indices in by.items():
            u,v=self.spans[name];f=factors[name]
            self.spans[name]=(append_span(u,f['U'][:,indices].to(u.device)),append_span(v,f['V'][:,indices].to(v.device)))
        self.rebuild()

    @torch.no_grad()
    def delta(self,name):
        m,u,v,_,_=self.items[name]
        d=m.weight.detach().float()-self.base[name].to(m.weight.device).float()
        return projected(d,u,v)

    @torch.no_grad()
    def stats(self):
        blocked=total=0.;ranks={}
        for name,(m,u,v,_,_) in self.items.items():
            ranks[name]={'output':u.shape[1],'input':v.shape[1]}
            if m.weight.grad is not None and (u.shape[1] or v.shape[1]):
                g=m.weight.grad.float();a=u.T@g;b=g@v;b=b-u@(u.T@b)
                blocked+=float(a.square().sum()+b.square().sum());total+=float(g.square().sum())
        return {'grad_blocked_share':blocked/total if total else 0.,'blocked_ranks':ranks}

    def save(self,path):
        torch.save({n:{'U':u.cpu(),'V':v.cpu()} for n,(u,v) in self.spans.items()},path)
