//-----------------------------------------------------------------------
// <copyright file="NeuralNet.cs" company="Seth Flowers">
//     All rights reserved.
// </copyright>
//-----------------------------------------------------------------------
namespace ArtificialNeuralNet
{
    using System;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;
    using System.Globalization;
    using System.Linq;

    /// <summary>
    /// <para>Represents an artificial neural net,</para>
    /// <para>containing layers of neurons, connected to other layers of neurons.</para>
    /// </summary>
    public class NeuralNet
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="NeuralNet"/> class.
        /// </summary>
        public NeuralNet()
            : this(neuronsPerLayer: new[] { 1 })
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NeuralNet" /> class.
        /// </summary>
        /// <param name="neuronsPerLayer">The neurons per layer.</param>
        /// <exception cref="System.ArgumentException">
        /// A neural net without layers is meaningless.;neuronsPerLayer
        /// or
        /// A neural net cannot have a layer with no inputs.;neuronsPerLayer
        /// </exception>
        public NeuralNet(int[] neuronsPerLayer)
        {
            if (neuronsPerLayer == null || neuronsPerLayer.Length == 0)
            {
                throw new ArgumentException("A neural net without layers is invalid.", "neuronsPerLayer");
            }
            else if (neuronsPerLayer.Any(n => n < 1))
            {
                throw new ArgumentException("A neural net cannot have a layer with no inputs.", "neuronsPerLayer");
            }

            // Instantiate the layers collection.
            this.Layers = new Collection<Layer>();

            // Create all the layers with the correct number of neurons per layer.
            foreach (int numberOfNeuronsInLayer in neuronsPerLayer)
            {
                this.Layers.Add(new Layer(numberOfNeuronsInLayer));
            }

            // Connect all the layers to each other.
            for (int i = 1; i < this.Layers.Count; i++)
            {
                Layer previousLayer = this.Layers[i - 1];
                Layer currentLayer = this.Layers[i];

                foreach (Neuron previousLayerNeuron in previousLayer.Neurons)
                {
                    foreach (Neuron currentLayerNeuron in currentLayer.Neurons)
                    {
                        Synapse synapse = new Synapse();
                        currentLayerNeuron.Inputs.Add(synapse);
                        previousLayerNeuron.Outputs.Add(synapse);
                    }
                }
            }

            // Add an output from each neuron in the output layer.
            foreach (Neuron neuron in this.Layers.Last().Neurons)
            {
                neuron.Outputs.Add(new Synapse());
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="NeuralNet" /> class.
        /// </summary>
        /// <param name="neuronsPerLayer">The neurons per layer.</param>
        /// <param name="weightsAndBiases">The weights and biases to initialize the net with.</param>
        /// <exception cref="System.ArgumentNullException">weightsAndBiases;Unable to initialize a neural net with null initialization data.</exception>
        /// <exception cref="System.ArgumentException">weightsAndBiases</exception>
        public NeuralNet(int[] neuronsPerLayer, double[] weightsAndBiases)
            : this(neuronsPerLayer)
        {
            if (weightsAndBiases == null)
            {
                throw new ArgumentNullException("weightsAndBiases", "Unable to initialize a neural net with null initialization data.");
            }

            // Determine how much initialization data we need by summing up all the inputs and biases
            // for all layers but the first input layer.
            int requiredInitializationCount = 0;

            // We do not initialize the bias for each node in the input layer, 
            // nor does the input layer have inputs.
            // We only need enough inputs for all the biases and inputs for every layer
            // other than the input layer.
            for (int i = 1; i < neuronsPerLayer.Length; i++)
            {
                // A bias is required for each neuron in the layer.
                requiredInitializationCount += neuronsPerLayer[i];

                // A synapse with a weight connects each node in the current layer to the previous layer.
                requiredInitializationCount += neuronsPerLayer[i] * neuronsPerLayer[i - 1];
            }

            if (weightsAndBiases.Length != requiredInitializationCount)
            {
                throw new ArgumentException(
                    string.Format(CultureInfo.CurrentCulture, "The total number of weights and biases to initialize the neural net is not the required amount of {0}.", requiredInitializationCount),
                    "weightsAndBiases");
            }

            int currentIndex = 0;

            // Initialize all the weights and biases for every 
            // input to every node in every layer but the first layer.
            foreach (Layer layer in this.Layers.Skip(1))
            {
                foreach (Neuron neuron in layer.Neurons)
                {
                    neuron.Bias = weightsAndBiases[currentIndex++];

                    foreach (Synapse synapse in neuron.Inputs)
                    {
                        synapse.Weight = weightsAndBiases[currentIndex++];
                    }
                }
            }
        }

        /// <summary>
        /// Gets the layers in this artificial neural net.
        /// </summary>
        /// <value>
        /// The layers in this artificial neural net.
        /// </value>
        public Collection<Layer> Layers { get; private set; }

        /// <summary>
        /// Runs the given input through this net, producing an output.
        /// </summary>
        /// <param name="inputs">The inputs.</param>
        /// <returns>Returns the outputs from processing the given input.</returns>
        /// <exception cref="System.ArgumentException">The number of inputs to a neural net should match the number of neurons in the input layer.;inputs</exception>
        public IEnumerable<double> Think(IList<double> inputs)
        {
            if (inputs == null || inputs.Count != this.Layers[0].Neurons.Count)
            {
                throw new ArgumentException("The number of inputs to a neural net should match the number of neurons in the input layer.", "inputs");
            }

            // Set the outputs of the input layer to be the inputs.
            for (int i = 0; i < inputs.Count; i++)
            {
                foreach (Synapse output in this.Layers[0].Neurons[i].Outputs)
                {
                    output.Value = inputs[i];
                }
            }

            // Let every layer after the input layer process the outputs from the previous layer.
            foreach (Layer layer in this.Layers.Skip(1))
            {
                layer.Think();
            }

            return
                from neuron in this.Layers.Last().Neurons
                from output in neuron.Outputs
                select output.Value;
        }
    }
}