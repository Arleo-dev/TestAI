using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestAI
{
    internal class NeuralNetwork
    {
        public Topology Topology { get;}
        public List<Layer> Layers { get; }

        public NeuralNetwork(Topology topology)
        {
            Topology = topology;
            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayer();
            CreateOutputLayer();
        }

        
        public Neuron FeedForward(params double[] inputSignals)
        {
            SendSignalsToInputNeuron(inputSignals);
            FeedForwardAllLayersAfterInput();
            if (Topology.OutputCount == 1)
            {
                return Layers.Last().Neurons[0];
            }
            else
            {
                return Layers.Last().Neurons.OrderByDescending(x => x.Output).First();
                
            }
        }
        public double Learn(Dictionary<double[], double> dataset, int epoch)
        {
            var error = 0.0;
            for(int i = 0; i < epoch; i++)
            {
                foreach (var item in dataset)
                {
                    error = Backpropagation(item.Value, item.Key);
                }
            }
            return error / error;
        }
        private double Backpropagation(double exprected, params double[] inputs)
        {
            var actiual = FeedForward(inputs).Output;
            var difference = actiual - exprected;
            foreach (var neuron in Layers.Last().Neurons)
            {
                neuron.Learn(difference, Topology.LearningRate);
            }
            for (int j = Layers.Count - 2; j >= 0 ; j--)
            { 
                var layer = Layers[j];
                var previouslyLayer = Layers[j + 1];
                for (int i = 0; i < layer.NeuronCount; i++)
                {
                    var neuron = layer.Neurons[i];
                    for (int k = 0; k < previouslyLayer.NeuronCount; k++)
                    {
                        var previousNeuron = previouslyLayer.Neurons[k];
                        var error = previousNeuron.Weights[i] * previousNeuron.Delta;
                        neuron.Learn(error, Topology.LearningRate);
                    }
                }
            }
            return difference * difference;
        }
        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayerSignals = Layers[i - 1].GetSignals();
                foreach (var neuron in layer.Neurons)
                {
                    neuron.FeedForward(previousLayerSignals);
                }
            }
        }
        private void SendSignalsToInputNeuron(params double[] inputSignals)
        {
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neuron = Layers[0].Neurons[i];
                neuron.FeedForward(signal);
            }
        }
        private void CreateInputLayer()
        {
            var inputNeurons = new List<Neuron>();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neuron = new Neuron(1, NeuronType.Input);
                inputNeurons.Add(neuron);
            }
            var inputLayer = new Layer(inputNeurons, NeuronType.Input);
            Layers.Add(inputLayer);
        }
        private void CreateHiddenLayer()
        {
            for (int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var lastLayer = Layers.Last();
                var hiddenNeurons = new List<Neuron>();
                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    var neuron = new Neuron(lastLayer.NeuronCount);
                    hiddenNeurons.Add(neuron);
                }
                var outputLayer = new Layer(hiddenNeurons, NeuronType.Output);
                Layers.Add(outputLayer);
            }
        }
        private void CreateOutputLayer()
        {
            var lastLayer = Layers.Last();
            var outputNeurons = new List<Neuron>();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neuron = new Neuron(lastLayer.NeuronCount, NeuronType.Output);
                outputNeurons.Add(neuron);
            }
            var outputLayer = new Layer(outputNeurons, NeuronType.Output);
            Layers.Add(outputLayer);
        }

    }
}
