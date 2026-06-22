import nengo
import nengo_spa as spa
import numpy as np
import memory_arch as ma

from importlib import reload

reload(ma)

d = 64

model = nengo.Network()
with model:
    # Testing Ports, Pairs, and Nodes objects
    
    vocab = spa.Vocabulary(d)
    vocab.add("NULL", np.zeros(d))

    ports = ma.Ports(vocab, 0.3, [], {}, [], [])

    key_idx = 0
    keys_to_tags = {}
   
    def make_cycler():
        state = 0
        sleeptime = 0.1
        stopwatch = 0.0
        tag_name = ""
        def cycler(t):
            global key_idx, keys_to_tags
            nonlocal state, sleeptime, stopwatch, tag_name
            if state == 0:
                print("current ports:", ports.ports_dict.keys())
                i = np.random.randint(0, len(ports.tags))
                tag_name = ports.tags[i]
                keys_to_tags[f"K_{key_idx}"] = tag_name
                print(f"anticipated next port: K_{key_idx} associated with {tag_name}")
                key_idx += 1
                state = 1
                stopwatch = t
            elif state == 1 and t > stopwatch + sleeptime:
                state = 2
                tag_name = "NULL"
            elif state == 2 and t > stopwatch + sleeptime * 2:
                stopwatch = 0.0
                state = 0
            return vocab[tag_name].v
        return cycler
    
    def value_getter():
        last = 0
        vec = np.random.normal(size=d, scale=1/np.sqrt(d))
        def rand_value(t):
            nonlocal last, vec
            i = int(t // 0.1)
            if i > last:
                vec[:] = np.random.normal(size=d, scale=1/np.sqrt(d))
            last = i
            return vec
        return rand_value

    
    tag_node = nengo.Node(output=make_cycler(), label="tag_node")

    value_node = nengo.Node(output=value_getter(), label="value_node")

    nengo.Connection(tag_node, ports.port_new[:d])
    nengo.Connection(value_node, ports.port_new[d:])

    def give_random_tag():
        current = ""
        def random_tag(t):
            nonlocal current
            beat = t*2
            if np.floor(beat) >= beat and np.floor(beat) < beat+0.1 and ports.ports_dict:
                current = np.random.choice(list(ports.ports_dict.keys()))
                print("fetching:", current)
            if current in ports.ports_dict:
                return np.concatenate((ports.ports_dict[current][0], vocab[current].v, ports.ports_dict[current][1]))
            return np.zeros(ports.dim*3)
        return random_tag

    def vcos_node(t, x):
        u = x[:d]
        v = x[d:]
        return vcos(u, v)

    def vcos(u, v):
        mag = np.sqrt(u@u) * np.sqrt(v@v)
        if mag == 0.0:
            return 0.0
        return (u@v)/mag

    random_tags = nengo.Node(output=give_random_tag(), label="random_tags")
    comp_tag = nengo.Node(output=vcos_node, size_in=d*2, label="comp_tag")
    nengo.Connection(random_tags[:d], comp_tag[:d])
    nengo.Connection(ports.port_tag, comp_tag[d:])
    nengo.Connection(random_tags[d:d*2], ports.port_tag)

    comp_val = nengo.Node(output=vcos_node, size_in=d*2, label="comp_val")
    nengo.Connection(random_tags[d*2:], comp_val[:d])
    nengo.Connection(ports.port_val, comp_val[d:])
    nengo.Connection(random_tags[d:d*2], ports.port_val)

    def make_random_keys():
        state = 0
        stopwatch = 0.0
        sleeptime = 1.0
        keys = np.zeros(2*d+1)
        def random_keys(t):
            nonlocal state, stopwatch, sleeptime, keys

            if state == 0 and ports.ports_dict:
                state = 1
                stopwatch = t
                key1_name = np.random.choice(list(ports.ports_dict.keys()))
                key2_name = np.random.choice(list(ports.ports_dict.keys()))

                keys[:d] = vocab[key1_name].v
                keys[d:2*d] = vocab[key2_name].v

                tag1 = keys_to_tags[key1_name]
                tag2 = keys_to_tags[key2_name]

                tag1_idx = ports.tags.index(tag1)
                tag2_idx = ports.tags.index(tag2)

                print(f"similarity 1: {vcos(vocab[tag1].v, ports.ports_dict[key1_name][0])}")
                print(f"similarity 2: {vcos(vocab[tag2].v, ports.ports_dict[key2_name][0])}")

                print(f"key 1: {key1_name}, tag 1: {keys_to_tags[key1_name]}, index: {tag1_idx}")
                print(f"key 2: {key2_name}, tag 2: {keys_to_tags[key2_name]}, index: {tag2_idx}")

                keys[-1] = (int(tag2_idx > tag1_idx))
            elif state == 1 and t > stopwatch + sleeptime:
                state = 0
                stopwatch = 0.0
                
            return keys
        return random_keys


    print(ports.tags)

    swap_node = nengo.Node(output=make_random_keys(), size_out=2*d+1, label="swap_node")
    should_swap_test = nengo.Node(output=lambda t, x: x[0] == x[1], size_in=2, label="sst")
    nengo.Connection(swap_node[:2*d], ports.port_swap)
    nengo.Connection(swap_node[-1], should_swap_test[0])
    nengo.Connection(ports.port_swap, should_swap_test[1])
