__author__ = 'mikhail'



import numpy

from scipy.misc import logsumexp





def logdotexp_vec_mat(loga, logM):
     return numpy.array([logsumexp(loga + x) for x in logM.T], copy=False)

def logdotexp_mat_vec(logM, logb):
    return numpy.array([logsumexp(x + logb) for x in logM], copy=False)

def flatten(x):
    a = []
    for y in x: a.extend(flatten(y) if isinstance(y, list) else [y])
    return a

class FeatureVector(object):
    def __init__(self, features, xlist, ylist=None):
        '''intermediates of features (sufficient statistics like)'''
        flist = features.features_edge
        glist = features.features
        self.K = len(features.labels)

              # expectation of features under empirical distribution (if ylist is specified)
        if ylist:
            self.Fss = numpy.zeros(features.size(), dtype=int)
            for y1, y2 in zip(["start"] + ylist, ylist + ["end"]):
                self.Fss[:len(flist)] += [f(y1, y2) for f in flist]
            for y1, x1 in zip(ylist, xlist):
                self.Fss[len(flist):] += [g(x1, y1) for g in glist]

        # index list of ON values of edge features
        self.Fon = [] # (n, #f, indexes)

        # for calculation of M_i
        self.Fmat = [] # (n, K, #f, K)-matrix
        self.Gmat = [] # (n, #g, K)-matrix
        for x in xlist:
            mt = numpy.zeros((len(glist), self.K), dtype=int)
            for j, g in enumerate(glist):
                mt[j] = [g(x, y) for y in features.labels]  # sparse
            self.Gmat.append(mt)
            #self._calc_fmlist(flist, x) # when fmlist depends on x_i (if necessary)

        # when fmlist doesn't depend on x_i
        self._calc_fmlist(features)


    def _calc_fmlist(self, features):
        flist = features.features_edge
        fmlist = []
        f_on = [[] for f in flist]
        for k1, y1 in enumerate(features.labels):
            mt = numpy.zeros((len(flist), self.K), dtype=int)
            for j, f in enumerate(flist):
                mt[j] = [f(y1, y2) for y2 in features.labels]  # sparse
                f_on[j].extend([k1 * self.K + k2 for k2, v in enumerate(mt[j]) if v == 1])
            fmlist.append(mt)
        self.Fmat.append(fmlist)
        self.Fon.append(f_on)

    def cost(self, theta):
        return numpy.dot(theta, self.Fss)

    def logMlist(self, theta_f, theta_g):
        '''for independent fmlists on x_i'''
        fv = numpy.zeros((self.K, self.K))
        for j, fm in enumerate(self.Fmat[0]):
            fv[j] = numpy.dot(theta_f, fm)
        return [fv + numpy.dot(theta_g, gm) for gm in self.Gmat] + [fv]

class Features(object):
    def __init__(self, labels):
        self.features = []
        self.features_edge = []
        self.labels = ["start","stop"] + flatten(labels)

    def start_label_index(self):
        return 0
    def stop_label_index(self):
        return 1
    def size(self):
        return len(self.features) + len(self.features_edge)
    def size_edge(self):
        return len(self.features_edge)
    def id2label(self, list):
        return [self.labels[id] for id in list]

    def add_feature(self, f):
        self.features.append(f)
    def add_feature_edge(self, f):
        self.features_edge.append(f)

class CRF(object):
    def __init__(self, features, regularity, sigma=1):
        self.features = features
        if regularity == 0:
            self.regularity = lambda w:0
            self.regularity_deriv = lambda w:0
        elif regularity == 1:
            self.regularity = lambda w:numpy.sum(numpy.abs(w)) / sigma
            self.regularity_deriv = lambda w:numpy.sign(w) / sigma
        else:
            v = sigma ** 2
            v2 = v * 2
            self.regularity = lambda w:numpy.sum(w ** 2) / v2
            self.regularity_deriv = lambda w:numpy.sum(w) / v

    def random_param(self):
        return numpy.random.randn(self.features.size())

    def logalphas(self, Mlist):
        logalpha = Mlist[0][self.features.start_label_index()] # alpha(1)
        logalphas = [logalpha]
        for logM in Mlist[1:]:
            logalpha = logdotexp_vec_mat(logalpha, logM)
            logalphas.append(logalpha)
        return logalphas

    def logbetas(self, Mlist):
        logbeta = Mlist[-1][:, self.features.stop_label_index()]
        logbetas = [logbeta]
        for logM in Mlist[-2::-1]:
            logbeta = logdotexp_mat_vec(logM, logbeta)
            logbetas.append(logbeta)
        return logbetas[::-1]

    def likelihood(self, fvs, theta):
        '''conditional log likelihood log p(Y|X)'''
        n_fe = self.features.size_edge() # number of features on edge
        t1, t2 = theta[:n_fe], theta[n_fe:]
        stop_index = self.features.stop_label_index()

        likelihood = 0
        for fv in fvs:
            logMlist = fv.logMlist(t1, t2)
            logZ = self.logalphas(logMlist)[-1][stop_index]
            likelihood += fv.cost(theta) - logZ
        return likelihood - self.regularity(theta)

    def gradient_likelihood(self, fvs, theta):
        n_fe = self.features.size_edge() # number of features on edge
        t1, t2 = theta[:n_fe], theta[n_fe:]
        stop_index = self.features.stop_label_index()
        start_index = self.features.start_label_index()

        grad = numpy.zeros(self.features.size())
        for fv in fvs:
            logMlist = fv.logMlist(t1, t2)
            logalphas = self.logalphas(logMlist)
            logbetas = self.logbetas(logMlist)
            logZ = logalphas[-1][stop_index]

            grad += numpy.array(fv.Fss, dtype=float) # empirical expectation

            expect = numpy.zeros_like(logMlist[0])
            for i in range(len(logMlist)):
                if i == 0:
                    expect[start_index] += numpy.exp(logalphas[i] + logbetas[i+1] - logZ)
                elif i < len(logbetas) - 1:
                    a = logalphas[i-1][:, numpy.newaxis]
                    expect += numpy.exp(logMlist[i] + a + logbetas[i+1] - logZ)
                else:
                    expect[:, stop_index] += numpy.exp(logalphas[i-1] + logbetas[i] - logZ)
            for k, indexes in enumerate(fv.Fon[0]):
                grad[k] -= numpy.sum(expect.take(indexes))

            for i, gm in enumerate(fv.Gmat):
                p_yi = numpy.exp(logalphas[i] + logbetas[i+1] - logZ)
                grad[n_fe:] -= numpy.sum(gm * numpy.exp(logalphas[i] + logbetas[i+1] - logZ), axis=1)

        return grad - self.regularity_deriv(theta)

    def inference(self, fvs, theta):
        from scipy import optimize
        likelihood = lambda x:-self.likelihood(fvs, x)
        likelihood_deriv = lambda x:-self.gradient_likelihood(fvs, x)
        return optimize.fmin_bfgs(likelihood, theta, fprime=likelihood_deriv)

    def tagging(self, fv, theta):
        n_fe = self.features.size_edge() # number of features on edge
        logMlist = fv.logMlist(theta[:n_fe], theta[n_fe:])

        logalphas = self.logalphas(logMlist)
        logZ = logalphas[-1][self.features.stop_label_index()]

        delta = logMlist[0][self.features.start_label_index()]
        argmax_y = []
        for logM in logMlist[1:]:
            h = logM + delta[:, numpy.newaxis]
            argmax_y.append(h.argmax(0))
            delta = h.max(0)
        Y = [delta.argmax()]
        for a in reversed(argmax_y):
            Y.append(a[Y[-1]])

        return Y[0] - logZ, Y[::-1]

    def tagging_verify(self, fv, theta):
        '''verification of tagging'''
        n_fe = self.features.size_edge() # number of features on edge
        logMlist = fv.logMlist(theta[:n_fe], theta[n_fe:])
        N = len(logMlist) - 1
        K = logMlist[0][0].size

        ylist = [0] * N
        max_p = -1e9
        argmax_p = None
        while True:
            logp = logMlist[0][self.features.start_label_index(), ylist[0]]
            for i in range(len(ylist)-1):
                logp += logMlist[i+1][ylist[i], ylist[i+1]]
            logp += logMlist[N][ylist[N-1], self.features.stop_label_index()]
            print ylist, logp
            if max_p < logp:
                max_p = logp
                argmax_p = ylist[:]

            for k in range(N-1,-1,-1):
                if ylist[k] < K-1:
                    ylist[k] += 1
                    break
                ylist[k] = 0
            else:
                break
        return max_p, argmax_p



def main():
    def load_data(data):
        texts = []
        labels = []
        text = []
        data = "\n" + data + "\n"
        for line in data.split("\n"):
            line = line.strip()
            if len(line) == 0:
                if len(text)>0:
                    texts.append(text)
                    labels.append(label)
                text = []
                label = []
            else:
                token, info, chunk = line.split()
                text.append((token, info))
                label.append(chunk)
        return (texts, labels)

    texts, labels = load_data("""
    The DT O
    initial JJ O
    report NN O
    of IN O
    the DT O
    Selenium NNP B-Dr
    and CC O
    Vitamin NNP B-Dr
    E NNP I-Dr
    Cancer NNP O
    Prevention NNP O
    Trial NN O
    ( ( O
    SELECT NNP O
    ) ) O
    found VBD O
    no DT O
    reduction NN O
    in IN O
    risk NN O
    of IN O
    prostate NN O
    cancer NN O
    with IN O
    either DT O
    selenium NN B-Dr
    or CC O
    vitamin NN B-Dr
    E NN I-Dr
    supplements NNS O
    but CC O
    a DT O
    statistically RB O
    nonsignificant JJ O
    increase NN O
    in IN O
    prostate NN O
    cancer NN O
    risk NN O
    with IN O
    vitamin NNP B-Dr
    E NNP I-Dr
    . . O
    Longer JJR O
    follow-up NN O
    and CC O
    more JJR O
    prostate NN O
    cancer NN O
    events NNS O
    provide VBP O
    further JJ O
    insight NN O
    into IN O
    the DT O
    relationship NN O
    of IN O
    vitamin JJ B-Dr
    E NN I-Dr
    and CC O
    prostate NN O
    cancer NN O
    . . O
    To TO O
    determine VB O
    the DT O
    long-term JJ O
    effect NN O
    of IN O
    vitamin NN B-Dr
    E NN I-Dr
    and CC O
    selenium NN B-Dr
    on IN O
    risk NN O
    of IN O
    prostate NN O
    cancer NN O
    in IN O
    relatively RB O
    healthy JJ O
    men NNS O
    . . O
    Oral JJ O
    selenium NN B-Dr
    ( ( O
    200 CD O
    IDg NNS O
    : : O
    d SYM O
    from IN O
    L-selenomethionine NNP O
    ) ) O
    with IN O
    matched VBN O
    vitamin NN B-Dr
    E NN I-Dr
    placebo NN I-Dr
    , , O
    vitamin VBZ B-Dr
    E NN I-Dr
    ( ( O
    400 CD O
    IU d CD O
    of IN O
    all DT O
    rac-I JJ O
    + JJ O
    - : O
    tocopheryl NN O
    acetate NN O
    ) ) O
    with IN O
    matched VBN O
    selenium NN B-Dr
    placebo NN I-Dr
    , , O
    both DT O
    agents NNS O
    , , O
    or CC O
    both DT O
    matched VBN O
    placebos NNS O
    for IN O
    a DT O
    planned JJ O
    follow-up NN O
    of IN O
    a DT O
    minimum NN O
    of IN O
    7 CD O
    and CC O
    maximum NN O
    of IN O
    12 CD O
    years NNS O
    . . O
    Compared VBN O
    with IN O
    the DT O
    placebo NN B-Dr
    ( ( O
    referent JJ O
    group NN O
    ) ) O
    in IN O
    which WDT O
    529 CD O
    men NNS O
    developed VBD O
    prostate NN O
    cancer NN O
    , , O
    620 JJ O
    men NNS O
    in IN O
    the DT O
    vitamin NN B-Dr
    E NN I-Dr
    group NN O
    developed VBD O
    prostate NN O
    cancer NN O
    ( ( O
    hazard NN O
    ratio NN O
    [ NNP O
    HR NNP O
    ] NNP O
    , , O
    1.17 CD O
    ; : O
    99 CD O
    % NN O
    CI NNP O
    , , O
    1.004-1 CD O
    .36 CD O
    , , O
    P NNP O
    = SYM O
    .008 CD O
    ) ) O
    ; : O
    as IN O
    did VBD O
    575 NNP O
    in IN O
    the DT O
    selenium NN B-Dr
    group NN O
    , , O
    and CC O
    555 CD O
    in IN O
    the DT O
    selenium NN B-Dr
    plus CC O
    vitamin NNP B-Dr
    E NNP I-Dr
    group NN O
    . . O
    Dietary JJ O
    supplementation NN O
    with IN O
    vitamin NN B-Dr
    E NN I-Dr
    significantly RB O
    increased VBD O
    the DT O
    risk NN O
    of IN O
    prostate NN O
    cancer NN O
    among IN O
    healthy JJ O
    men NNS O
    . . O
    The DT O
    primary JJ O
    analysis NN O
    included VBD O
    34,887 CD O
    men NNS O
    who WP O
    were VBD O
    randomly RB O
    assigned VBN O
    to TO O
    1 CD O
    of IN O
    4 CD O
    treatment NN O
    groups NNS O
    : : O
    8752 CD O
    to TO O
    receive VB O
    selenium NN B-Dr
    ; : O
    8737 CD O
    , , O
    vitamin NNP B-Dr
    E NNP I-Dr
    ; : O
    8702 CD O
    , , O
    both DT O
    agents NNS O
    , , O
    and CC O
    8696 CD O
    , , O
    placebo NN B-Dr
    . . O
    Compared VBN O
    with IN O
    placebo NN B-Dr
    , , O
    the DT O
    absolute JJ O
    increase NN O
    in IN O
    risk NN O
    of IN O
    prostate NN O
    cancer NN O
    per IN O
    1000 CD O
    person-years NNS O
    was VBD O
    1.6 CD O
    for IN O
    vitamin NN B-Dr
    E NN I-Dr
    , , O
    0.8 CD O
    for IN O
    selenium NN B-Dr
    , , O
    and CC O
    0.4 CD O
    for IN O
    the DT O
    combination NN O
    . . O
    We PRP O
    randomly RB O
    assigned VBN O
    3222 CD O
    women NNS O
    with IN O
    HER2-positive VBG O
    early-stage JJ O
    breast NN O
    cancer NN O
    to TO O
    receive VB O
    doxorubicin NN B-Dr
    and CC O
    cyclophosphamide NN B-Dr
    followed VBD O
    by IN O
    docetaxel NN B-Dr
    every DT O
    3 CD O
    weeks NNS O
    ( ( O
    AC-T NNP O
    ) ) O
    , , O
    the DT O
    same JJ O
    regimen NN O
    plus CC O
    52 CD O
    weeks NNS O
    of IN O
    trastuzumab NN B-Dr
    ( ( O
    AC-T NNS O
    plus IN O
    trastuzumab NN B-Dr
    ) ) O
    , , O
    or CC O
    docetaxel NN B-Dr
    and CC O
    carboplatin NN B-Dr
    plus CC O
    52 CD O
    weeks NNS O
    of IN O
    trastuzumab NN B-Dr
    ( ( O
    TCH NNP O
    ) ) O
    . . O
    Trastuzumab NNP B-Dr
    improves VBZ O
    survival NN O
    in IN O
    the DT O
    adjuvant JJ O
    treatment NN O
    of IN O
    HER-positive JJ O
    breast NN O
    cancer NN O
    , , O
    although IN O
    combined VBN O
    therapy NN O
    with IN O
    anthracycline-based JJ O
    regimens NNS O
    has VBZ O
    been VBN O
    associated VBN O
    with IN O
    cardiac JJ O
    toxicity NN O
    . . O
    We PRP O
    wanted VBD O
    to TO O
    evaluate VB O
    the DT O
    efficacy NN O
    and CC O
    safety NN O
    of IN O
    a DT O
    new JJ O
    nonanthracycline NN O
    regimen NN O
    with IN O
    trastuzumab NN B-Dr
    . . O
    The DT O
    estimated JJ O
    disease-free JJ O
    survival NN O
    rates NNS O
    at IN O
    5 CD O
    years NNS O
    were VBD O
    75 CD O
    % NN O
    among IN O
    patients NNS O
    receiving VBG O
    AC-T NNP O
    , , O
    84 CD O
    % NN O
    among IN O
    those DT O
    receiving NN O
    AC-T NNP O
    plus CC O
    trastuzumab NNP B-Dr
    , , O
    and CC O
    81 CD O
    % NN O
    among IN O
    those DT O
    receiving VBG O
    TCH NNP O
    . . O
    No DT O
    significant JJ O
    differences NNS O
    in IN O
    efficacy NN O
    ( ( O
    disease-free NN O
    or CC O
    overall JJ O
    survival NN O
    ) ) O
    were VBD O
    found VBN O
    between IN O
    the DT O
    two CD O
    trastuzumab NN B-Dr
    regimens NNS O
    , , O
    whereas IN O
    both DT O
    were VBD O
    superior JJ O
    to TO O
    AC-T VB O
    . . O
    The DT O
    rates NNS O
    of IN O
    congestive JJ O
    heart NN O
    failure NN O
    and CC O
    cardiac JJ O
    dysfunction NN O
    were VBD O
    significantly RB O
    higher JJR O
    in IN O
    the DT O
    group NN O
    receiving VBG O
    AC-T JJ O
    plus IN O
    trastuzumab NN B-Dr
    than IN O
    in IN O
    the DT O
    TCH NNP O
    group NN
    ( ( O
    P NNP O
    0.001 CD O
    ) ) O
    . . O
    The DT O
    addition NN O
    of IN O
    1 CD O
    year NN O
    of IN O
    adjuvant JJ O
    trastuzumab NN B-Dr
    significantly RB O
    improved VBD O
    disease-free JJ O
    and CC O
    overall JJ O
    survival NN O
    among IN O
    women NNS O
    with IN O
    HER2-positive NN O
    breast NN O
    cancer NN O
    . . O
    The DT O
    risk-benefit JJ O
    ratio NN O
    favored VBD O
    the DT O
    nonanthracycline JJ O
    TCH NNP O
    regimen NN O
    over IN O
    AC-T NNP O
    plus CC O
    trastuzumab NNP B-Dr
    , , O
    given VBN O
    its PRP$ O
    similar JJ O
    efficacy NN O
    , , O
    fewer RBR O
    acute JJ O
    toxic JJ O
    effects NNS O
    , , O
    and CC O
    lower JJR O
    risks NNS O
    of IN O
    cardiotoxicity NN O
    and CC O
    leukemia NN O
    . . O
    """)

    test_texts, test_labels = load_data("""
    After IN O
    completing VBG O
    a DT O
    regimen NN O
    of IN O
    14 CD O
    to TO O
    21 CD O
    days NNS O
    of IN O
    parenteral JJ O
    acyclovir NN B-Dr
    , , O
    the DT O
    infants NNS O
    were VBD O
    randomly RB O
    assigned VBN O
    to TO O
    immediate JJ O
    acyclovir NN B-Dr
    suppression NN O
    ( ( O
    300 CD O
    mg NN O
    per IN O
    square JJ O
    meter NN O
    of IN O
    body-surface JJ O
    area NN O
    per IN O
    dose NN O
    orally NN O
    , , O
    three CD O
    times NNS O
    daily RB O
    for IN O
    6 CD O
    months NNS O
    ) ) O
    or CC O
    placebo NN B-Dr
    . . O
    """)

    features = Features(labels)
    tokens = dict([(i[0],1) for x in texts for i in x]).keys()
    infos = dict([(i[1],1) for x in texts for i in x]).keys()

    for label in features.labels:
        for token in tokens:
            features.add_feature( lambda x, y, l=label, t=token: 1 if y==l and x[0]==t else 0 )
        for info in infos:
            features.add_feature( lambda x, y, l=label, i=info: 1 if y==l and x[1]==i else 0 )
    features.add_feature_edge( lambda y_, y: 0 )

    fvs = [FeatureVector(features, x, y) for x, y in zip(texts, labels)]
    fv = fvs[0]
    text_fv = FeatureVector(features, test_texts[0]) # text sequence without labels


    crf = CRF(features, 2)
    theta = crf.random_param()

    print "features:", features.size()
    print "labels:", len(features.labels)

    #print "theta:", theta
    print "log likelihood:", crf.likelihood(fvs, theta)
    prob, ys = crf.tagging(text_fv, theta)
    print "tagging:", prob, features.id2label(ys)

    theta = crf.inference(fvs, theta)

    #print "theta:", theta
    print "log likelihood:", crf.likelihood(fvs, theta)
    prob, ys = crf.tagging(text_fv, theta)
    print "tagging:", prob, zip(test_texts[0], test_labels[0], features.id2label(ys))

if __name__ == "__main__":
    main()
