using TestAI;
var topology = new Topology(8, 1, 0.00001, 8);
var network = new NeuralNetwork(topology);
var data = new Dictionary<double[], double>();

int i = 0;
double val = 0;
double[] vs = new double[256];
while (val <= 1.000001)
{
    vs[i] = val;
    i++;
    val += 0.00392;
}

for (int j = 0; j < 256; j++)
{
    data.Add(Decimaltobinar(j), j);
}


var difference = network.Learn(data, 10000);

foreach (var item in data)
{
    var res = network.FeedForward(item.Key).Output;
    Console.WriteLine($"dif = {difference}\t {item.Value:f5}   res = {res:f5}");
}

var res1 = network.FeedForward(0, 0, 0, 0, 0, 0, 0, 0).Output;

Console.WriteLine($"out {res1}");

static double[] Decimaltobinar(int n)
{
    int i;
    double[] a = new double[8];
    for (i = 0; n > 0; i++)
    {
        a[i] = n % 2;
        n = n / 2;
    }
    Array.Reverse(a);
    return a;
}