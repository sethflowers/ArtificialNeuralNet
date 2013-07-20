﻿//-----------------------------------------------------------------------
// <copyright file="NeuralNet.cs" company="Seth Flowers">
//     All rights reserved.
// </copyright>
//-----------------------------------------------------------------------
namespace ArtificialNeuralNet
{
    using System;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;
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

            // Add an input to each neuron in the input layer.
            foreach (Neuron neuron in this.Layers.First().Neurons)
            {
                neuron.Inputs.Add(new Synapse());
            }

            // Add an output from each neuron in the output layer.
            foreach (Neuron neuron in this.Layers.Last().Neurons)
            {
                neuron.Outputs.Add(new Synapse());
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

            for (int i = 0; i < inputs.Count; i++)
            {
                this.Layers[0].Neurons[i].Inputs[0].Value = inputs[i];
            }

            foreach (Layer layer in this.Layers)
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